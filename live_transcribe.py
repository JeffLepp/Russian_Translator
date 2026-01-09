import json
import queue
import sys
import threading
import subprocess
import time
import re

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from argostranslate.translate import translate

# -------- Audio config --------
SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)

# -------- Models --------
MODEL_RU = "Models/vosk-model-small-ru-0.22"
MODEL_EN = "Models/vosk-model-small-en-us-0.15"  # change to your EN model folder

# -------- VAD-ish thresholds --------
SILENCE_RMS = 250
END_SILENCE_MS = 1250

# Push-to-talk behavior
PTT_SILENCE_GRACE_MS = 250  # small grace to avoid chopping endings

audio_q: queue.Queue[bytes] = queue.Queue()

# Space key state for pressing space
ptt_active = threading.Event()      # toggled ON/OFF by space press


def cb(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(bytes(indata))


def rms_int16_bytes(b: bytes) -> float:
    x = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def parse_text(result_json: str):
    try:
        r = json.loads(result_json)
    except Exception:
        return "", []
    text = (r.get("text") or "").strip()
    words = r.get("result") or []
    return text, words


def avg_conf(words) -> float:
    confs = [w.get("conf", 0.0) for w in words if isinstance(w, dict)]
    return (sum(confs) / len(confs)) if confs else 0.0

def tts_text_cleanup(text: str) -> str:
    text = text.strip()
    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)
    # Replace problematic punctuation for espeak
    text = text.replace("—", ", ").replace("–", ", ")
    text = text.replace("…", ".")
    # Optional: avoid reading quotes/brackets weirdly
    text = text.replace("«", "").replace("»", "")
    text = re.sub(r"[\[\]{}<>]", "", text)
    return text

# This is legacy TTS function
def tts_ru(text: str):
    if not text.strip():
        return
    text = tts_text_cleanup(text)

    try:
        subprocess.run(
            ["espeak-ng", "-v", "ru+f3", "-s", "155", "-p", "55", "-a", "170", text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("TTS missing: install espeak-ng.", file=sys.stderr)

# This is the currently used TTS function
def speak_chunks_espeak(text: str, max_len: int = 180):
    text = tts_text_cleanup(text)
    if not text:
        return

    parts = []
    cur = ""
    for token in text.split(" "):
        if len(cur) + 1 + len(token) > max_len:
            if cur:
                parts.append(cur)
            cur = token
        else:
            cur = (cur + " " + token).strip()
    if cur:
        parts.append(cur)

    for p in parts:
        subprocess.run(
            [
                "espeak-ng",      # TTS engine executable
                "-v", "ru+f3",    # voice: Russian (+f3 = female variant)
                "-s", "120",      # speed: words per minute (140–170 is natural)
                "-p", "25",       # pitch: mid-range, avoids robotic tone
                "-a", "120",      # amplitude: volume (100–200 typical)
                p                # chunk to speak (IMPORTANT: use p, not text)
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def start_space_listener():
    """
    Toggle SPACE: press once to start EN, press again to stop EN.
    """
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                # toggle
                if ptt_active.is_set():
                    ptt_active.clear()
                else:
                    ptt_active.set()
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)  # no on_release
    listener.daemon = True
    listener.start()
    return listener

def main():
    print("Loading RU model...")
    m_ru = Model(MODEL_RU)
    rec_ru = KaldiRecognizer(m_ru, SAMPLE_RATE)

    print("Loading EN model...")
    m_en = Model(MODEL_EN)
    rec_en = KaldiRecognizer(m_en, SAMPLE_RATE)

    # Start keyboard UI
    start_space_listener()
    print("UI: hold SPACE to push-to-talk in English (release to translate EN->RU + speak).")
    print("Default: listens for Russian, prints RU->EN on utterance end.")
    print("Listening (Ctrl+C to stop).")

    # RU state
    ru_in_speech = False
    ru_silence_ms = 0

    # EN push-to-talk state
    en_active = False
    en_last_voice_ms = 0  # for a tiny grace period

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SAMPLES,
        dtype="int16",
        channels=1,
        callback=cb,
    ):
        while True:
            b = audio_q.get()
            level = rms_int16_bytes(b)
            is_voice = level >= SILENCE_RMS

            # -------- Push-to-talk EN path (SPACE held) --------
            if ptt_active.is_set():
                # While holding space, we prioritize EN recognition.
                if not en_active:
                    en_active = True
                    rec_en.Reset()
                    en_last_voice_ms = 0
                    # Optional: visual cue
                    print("\n[PTT EN] Listening...", flush=True)

                rec_en.AcceptWaveform(b)
                if is_voice:
                    en_last_voice_ms = 0
                else:
                    en_last_voice_ms += BLOCK_MS

                # While EN PTT is active, we do NOT advance RU utterance state
                # (prevents the two recognizers competing for the same audio).
                continue

            # If we just released space, finalize EN and speak RU translation
            if en_active and not ptt_active.is_set():
                # Small grace: if user released during quiet gap, still accept a few blocks
                # (helps catch trailing consonants). This is optional.
                grace_blocks = int(PTT_SILENCE_GRACE_MS / BLOCK_MS)
                for _ in range(grace_blocks):
                    try:
                        b2 = audio_q.get_nowait()
                        rec_en.AcceptWaveform(b2)
                    except queue.Empty:
                        break

                en_final = rec_en.FinalResult()
                en_text, en_words = parse_text(en_final)
                en_conf = avg_conf(en_words)

                if en_text:
                    ru_out = translate(en_text, "en", "ru")
                    print(f"[FINAL en {en_conf:.2f}] {en_text}")
                    print(f"[EN->RU] {ru_out}")
                    # tts_ru(ru_out)
                    speak_chunks_espeak(ru_out, 500)
                else:
                    print("[PTT EN] (no text)")

                en_active = False
                # After PTT ends, we fall back to RU mode fresh
                ru_in_speech = False
                ru_silence_ms = 0
                rec_ru.Reset()
                continue

            # -------- Default RU path (hands-free) --------
            if is_voice:
                if not ru_in_speech:
                    ru_in_speech = True
                    ru_silence_ms = 0
                    rec_ru.Reset()
                rec_ru.AcceptWaveform(b)

            else:
                if ru_in_speech:
                    rec_ru.AcceptWaveform(b)
                    ru_silence_ms += BLOCK_MS

                    if ru_silence_ms >= END_SILENCE_MS:
                        ru_final = rec_ru.FinalResult()
                        ru_text, ru_words = parse_text(ru_final)
                        ru_conf = avg_conf(ru_words)

                        if ru_text:
                            en_out = translate(ru_text, "ru", "en")
                            print(f"[FINAL ru {ru_conf:.2f}] {en_out}")

                        ru_in_speech = False
                        ru_silence_ms = 0


if __name__ == "__main__":
    main()