import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import wave
import tempfile

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

# -----------------------------
# Piper TTS config (PIPER ONLY)
# -----------------------------
PIPER_LENGTH_SCALE = 0.95   # slower -> more natural; try 0.95..1.20
PIPER_NOISE_SCALE = 0.75    # variability; try 0.70..0.95
PIPER_NOISE_W = 0.65        # prosody timing; try 0.60..0.95

# -----------------------------
# Globals / state
# -----------------------------
audio_q: queue.Queue[bytes] = queue.Queue()
ptt_active = threading.Event()  # toggled by space press
tts_q: queue.Queue[str] = queue.Queue(maxsize=8)  # TTS worker queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_BIN = os.path.join(os.path.dirname(sys.executable), "piper")
PIPER_MODEL_PATH = os.path.join(
    BASE_DIR, "Models/piper/ru_RU/irina/medium/ru_RU-irina-medium.onnx"
)
# Piper expects sibling config named *.onnx.json
PIPER_CONFIG_PATH = PIPER_MODEL_PATH + ".json"

# Loaded at startup
PIPER_SAMPLE_RATE: int = 22050

STARTUP_TTS_TEST = (
    "Привет. Это тест синтеза речи. "
    "Если ты слышишь это, значит всё работает нормально, "
    "и голос загружен корректно."
)


# -----------------------------
# Startup checks
# -----------------------------

def piper_supports_model_flag() -> bool:
    try:
        out = subprocess.run(
            [PIPER_BIN, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
        return "--model" in out
    except Exception:
        return False



def _fatal(msg: str, exit_code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(exit_code)


def _require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        _fatal(f"{label} not found: {path}")


def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        _fatal(f"Required binary not found in PATH: {name}")

def init_piper() -> None:
    global PIPER_SAMPLE_RATE

    _require_file(PIPER_BIN, "Piper TTS binary (.venv/bin/piper)")
    if not (shutil.which("pw-cat") or shutil.which("paplay") or shutil.which("aplay")):
        _fatal("No audio player found (need pw-cat or paplay or aplay)")


    _require_file(PIPER_MODEL_PATH, "Piper model")
    _require_file(PIPER_CONFIG_PATH, "Piper model config (.onnx.json)")

    with open(PIPER_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    PIPER_SAMPLE_RATE = (
        cfg.get("audio", {}).get("sample_rate")
        or cfg.get("sample_rate")
        or 22050
    )

    print("Using Piper:", PIPER_BIN, flush=True)
    print("Piper sample rate:", PIPER_SAMPLE_RATE, flush=True)


# -----------------------------
# Audio helpers
# -----------------------------
def audio_cb(indata, frames, time_info, status):
    if status:
        # keep it minimal; status goes to stderr
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

    # make MT output more speakable
    text = text.replace("—", ", ").replace("–", ", ").replace("…", ".")
    text = text.replace("«", "").replace("»", "")
    text = re.sub(r"[\[\]{}<>]", "", text)

    text = re.sub(
        r"\s+(и|но|а|однако)\s+",
        r", \1 ",
        text,
        flags=re.IGNORECASE
    )

    # ensure a natural stop
    if text and text[-1] not in ".!?":
        text += "."

    return text


def chunk_text(text: str, max_len: int = 140) -> list[str]:
    text = tts_text_cleanup(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, cur = [], ""

    def flush():
        nonlocal cur
        if cur.strip():
            parts.append(cur.strip())
        cur = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # If a sentence is too long, split on commas
        if len(s) > max_len:
            sub = [p.strip() for p in s.split(",") if p.strip()]
            for p in sub:
                p2 = p + ","
                if len(p2) > max_len:
                    # hard split as last resort
                    for i in range(0, len(p2), max_len):
                        parts.append(p2[i:i+max_len].strip())
                else:
                    parts.append(p2.strip())
            continue

        if len(cur) + 1 + len(s) > max_len:
            flush()
            cur = s
        else:
            cur = (cur + " " + s).strip()

    flush()
    return [p.rstrip(",") + ("" if p.endswith((".", "!", "?")) else ".") for p in parts]


def _pick_player():
    # PipeWire: pw-cat supports stdin streaming
    if shutil.which("pw-cat"):
        return (
            ["pw-cat", "--playback", "--rate", str(PIPER_SAMPLE_RATE), "--channels", "1", "--format", "s16le"],
            "raw",
        )

    # PulseAudio / PipeWire-Pulse (WAV file)
    if shutil.which("paplay"):
        return (["paplay"], "wav")

    # ALSA raw stdin
    return (["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(PIPER_SAMPLE_RATE), "-c", "1"], "raw")


# -----------------------------
# Piper TTS (streaming, no fallback)
# -----------------------------
def speak_piper_stream(text: str) -> None:
    text = tts_text_cleanup(text)
    if not text:
        return

    if piper_supports_model_flag():
        piper_cmd = [
            PIPER_BIN,
            "--model", PIPER_MODEL_PATH,
            "--config", PIPER_CONFIG_PATH,
            "--output-raw",
            "--length_scale", str(PIPER_LENGTH_SCALE),
            "--noise_scale", str(PIPER_NOISE_SCALE),
            "--noise_w", str(PIPER_NOISE_W),
        ]
    else:
        piper_cmd = [
            PIPER_BIN,
            "--model", PIPER_MODEL_PATH,
            "--config", PIPER_CONFIG_PATH,
            "--output-raw",
            "--length_scale", str(PIPER_LENGTH_SCALE),
            "--noise_scale", str(PIPER_NOISE_SCALE),
            "--noise_w", str(PIPER_NOISE_W),
        ]



    p = subprocess.Popen(
        piper_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    try:
        assert p.stdin is not None
        assert p.stdout is not None

        p.stdin.write((text + "\n").encode("utf-8"))
        p.stdin.flush()
        p.stdin.close()

        player_cmd, mode = _pick_player()

        if mode == "raw":
            a = subprocess.Popen(
                player_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                assert a.stdin is not None
                while True:
                    chunk = p.stdout.read(4096)
                    if not chunk:
                        break
                    a.stdin.write(chunk)
            finally:
                try:
                    a.stdin.close()
                except Exception:
                    pass
                try:
                    a.wait(timeout=10)
                except Exception:
                    try:
                        a.kill()
                    except Exception:
                        pass

        else:
            # mode == "wav" (paplay): write a tiny temp WAV then play it
            raw = p.stdout.read()
            if not raw:
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                wav_path = tf.name
            try:
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # int16
                    wf.setframerate(int(PIPER_SAMPLE_RATE))
                    wf.writeframes(raw)
                subprocess.run(player_cmd + [wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    except Exception:
        # drop utterance silently
        pass
    finally:
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def tts_worker():
    while True:
        text = tts_q.get()
        try:
            speak_piper_stream(text)
        except Exception:
            # drop utterance silently
            pass
        finally:
            tts_q.task_done()


def start_tts_worker():
    th = threading.Thread(target=tts_worker, daemon=True)
    th.start()
    return th


def say_ru(text: str):
    """
    Queue Russian speech. Drops oldest if full so output stays live.
    Uses chunking so Piper speaks naturally instead of word-by-word.
    """
    for p in chunk_text(text, max_len=160):
        if not p:
            continue
        try:
            tts_q.put_nowait(p)
        except queue.Full:
            try:
                _ = tts_q.get_nowait()
                tts_q.task_done()
            except queue.Empty:
                pass
            try:
                tts_q.put_nowait(p)
            except queue.Full:
                pass


# -----------------------------
# Keyboard / UI
# -----------------------------
def start_space_listener():
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

    say_ru(ru_out)


def handle_ru_utterance_end(rec_ru: KaldiRecognizer):
    ru_final = rec_ru.FinalResult()
    ru_text, ru_words = parse_text(ru_final)
    ru_conf = avg_conf(ru_words)

    if ru_text:
        en_out = translate(ru_text, "ru", "en")
        print(f"[FINAL ru {ru_conf:.2f}] {en_out}")


def main():
    init_piper()

    print("Loading RU model...")
    m_ru = Model(MODEL_RU)
    rec_ru = KaldiRecognizer(m_ru, SAMPLE_RATE)

    print("Loading EN model...")
    m_en = Model(MODEL_EN)
    rec_en = KaldiRecognizer(m_en, SAMPLE_RATE)

    start_space_listener()
    start_tts_worker()
    say_ru(STARTUP_TTS_TEST) ##### TTS STARTUP CHECK: OUTPUTS EXAMPLE AT STARTUP

    print("UI: press SPACE once to start EN PTT, press again to stop (EN->RU + speak).")
    print("Default: listens for Russian, prints RU->EN on utterance end.")
    print("Listening (Ctrl+C to stop).")

    ru_in_speech = False
    ru_silence_ms = 0
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

            # EN Push-to-talk
            if ptt_active.is_set():
                if not en_active:
                    en_active = True
                    rec_en.Reset()
                    print("\n[PTT EN] Listening...", flush=True)

                rec_en.AcceptWaveform(b)
                continue

            # Released PTT -> finalize EN and speak RU
            if en_active and not ptt_active.is_set():
                finalize_en_and_speak_ru(rec_en)
                en_active = False

                # reset RU state
                ru_in_speech = False
                ru_silence_ms = 0
                rec_ru.Reset()
                continue

            # Default RU hands-free
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
