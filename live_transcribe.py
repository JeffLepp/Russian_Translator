import json
import queue
import sys

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from argostranslate.translate import translate

SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)

MODEL_RU = "Models/vosk-model-small-ru-0.22"

SILENCE_RMS = 250
END_SILENCE_MS = 1250

audio_q: queue.Queue[bytes] = queue.Queue()


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


def main():
    print("Loading RU model...")
    m_ru = Model(MODEL_RU)
    rec_ru = KaldiRecognizer(m_ru, SAMPLE_RATE)

    in_speech = False
    silence_ms = 0

    print("Listening (Ctrl+C to stop).")
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

            if is_voice:
                if not in_speech:
                    in_speech = True
                    silence_ms = 0
                    rec_ru.Reset()
                rec_ru.AcceptWaveform(b)

            else:
                if in_speech:
                    rec_ru.AcceptWaveform(b)
                    silence_ms += BLOCK_MS

                    if silence_ms >= END_SILENCE_MS:
                        ru_final = rec_ru.FinalResult()
                        text, words = parse_text(ru_final)
                        conf = avg_conf(words)
                        en = translate(text, "ru", "en")

                        if text:
                            print(f"[FINAL ru {conf:.2f}] {en}")

                        in_speech = False
                        silence_ms = 0


if __name__ == "__main__":
    main()
