import json
import queue
import re
import subprocess
import sys
import threading

import numpy as np
import sounddevice as sd
from argostranslate.translate import translate
from vosk import KaldiRecognizer, Model

# -----------------------------
# Config
# -----------------------------
SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)

MODEL_RU = "Models/vosk-model-small-ru-0.22"
MODEL_EN = "Models/vosk-model-small-en-us-0.15"

SILENCE_RMS = 250
END_SILENCE_MS = 1250
PTT_SILENCE_GRACE_MS = 250

# espeak settings (fallback TTS)
ESPEAK_VOICE = "ru+f3"
ESPEAK_SPEED = "120"
ESPEAK_PITCH = "25"
ESPEAK_AMP = "120"

# -----------------------------
# Globals / state
# -----------------------------
audio_q: queue.Queue[bytes] = queue.Queue()
ptt_active = threading.Event()  # toggled by space press


# -----------------------------
# Audio helpers
# -----------------------------
def audio_cb(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(bytes(indata))


def rms_int16_bytes(b: bytes) -> float:
    x = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


# -----------------------------
# Vosk helpers
# -----------------------------
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


# -----------------------------
# Text / TTS helpers
# -----------------------------
def tts_text_cleanup(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("—", ", ").replace("–", ", ").replace("…", ".")
    text = text.replace("«", "").replace("»", "")
    text = re.sub(r"[\[\]{}<>]", "", text)
    return text


def chunk_text(text: str, max_len: int = 180) -> list[str]:
    text = tts_text_cleanup(text)
    if not text:
        return []
    parts: list[str] = []
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
    return parts


def speak_espeak(text: str, max_len: int = 180):
    parts = chunk_text(text, max_len=max_len)
    for p in parts:
        subprocess.run(
            [
                "espeak-ng",              # TTS engine executable
                "-v", ESPEAK_VOICE,       # voice: Russian (+f3 = female variant)
                "-s", ESPEAK_SPEED,       # speed: words per minute
                "-p", ESPEAK_PITCH,       # pitch: 0-99
                "-a", ESPEAK_AMP,         # amplitude: volume
                p,                        # chunk to speak
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# -----------------------------
# Keyboard / UI
# -----------------------------
def start_space_listener():
    """
    Toggle SPACE: press once to start EN, press again to stop EN.
    """
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                if ptt_active.is_set():
                    ptt_active.clear()
                else:
                    ptt_active.set()
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


# -----------------------------
# Main loop
# -----------------------------
def finalize_en_and_speak_ru(rec_en: KaldiRecognizer):
    # Grab a few extra blocks after release to avoid chopping endings
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

    if not en_text:
        print("[PTT EN] (no text)")
        return

    ru_out = translate(en_text, "en", "ru")
    print(f"[FINAL en {en_conf:.2f}] {en_text}")
    print(f"[EN->RU] {ru_out}")

    # Fallback TTS (swap to Piper later)
    speak_espeak(ru_out, max_len=180)


def handle_ru_utterance_end(rec_ru: KaldiRecognizer):
    ru_final = rec_ru.FinalResult()
    ru_text, ru_words = parse_text(ru_final)
    ru_conf = avg_conf(ru_words)

    if ru_text:
        en_out = translate(ru_text, "ru", "en")
        print(f"[FINAL ru {ru_conf:.2f}] {en_out}")


def main():
    print("Loading RU model...")
    m_ru = Model(MODEL_RU)
    rec_ru = KaldiRecognizer(m_ru, SAMPLE_RATE)

    print("Loading EN model...")
    m_en = Model(MODEL_EN)
    rec_en = KaldiRecognizer(m_en, SAMPLE_RATE)

    start_space_listener()
    print("UI: press SPACE once to start EN PTT, press again to stop (EN->RU + speak).")
    print("Default: listens for Russian, prints RU->EN on utterance end.")
    print("Listening (Ctrl+C to stop).")

    # RU state
    ru_in_speech = False
    ru_silence_ms = 0

    # EN PTT state
    en_active = False

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SAMPLES,
        dtype="int16",
        channels=1,
        callback=audio_cb,
    ):
        while True:
            b = audio_q.get()
            level = rms_int16_bytes(b)
            is_voice = level >= SILENCE_RMS

            # -----------------------------
            # EN Push-to-talk mode
            # -----------------------------
            if ptt_active.is_set():
                if not en_active:
                    en_active = True
                    rec_en.Reset()
                    print("\n[PTT EN] Listening...", flush=True)

                rec_en.AcceptWaveform(b)
                # While EN is active, ignore RU pipeline
                continue

            # We just released PTT -> finalize EN + speak RU
            if en_active and not ptt_active.is_set():
                finalize_en_and_speak_ru(rec_en)

                en_active = False
                # reset RU state after speaking
                ru_in_speech = False
                ru_silence_ms = 0
                rec_ru.Reset()
                continue

            # -----------------------------
            # Default RU hands-free mode
            # -----------------------------
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
                        handle_ru_utterance_end(rec_ru)
                        ru_in_speech = False
                        ru_silence_ms = 0


if __name__ == "__main__":
    main()
