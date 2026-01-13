# Real-Time RU ↔ EN Speech Translation System 

An offline, real-time speech translation pipeline combining automatic speech recognition (ASR),
neural machine translation (NMT), and neural text-to-speech (TTS). Designed for low-latency,
interactive use on consumer hardware without relying on cloud services.

- **Always listens for Russian** (hands-free) and prints **RU → EN** when you stop speaking.
- **Push-to-talk English mode** press space to speak English, translate EN → RU, and synthesize Russian speech using Piper TTS.

## Features

- Offline speech recognition with **Vosk**
- Offline machine translation with **Argos Translate**
- Offline TTS with **Piper**
- Push-to-talk English mode (SPACE)
- Hands-free Russian mode
- Built-in mic-mute during TTS playback to reduce self-listening feedback


## ML Architecture

Audio Input
→ Voice Activity Detection
→ ASR (Vosk)
→ NMT (Argos / Marian)
→ Neural TTS (Piper)
→ Audio Output

The system prioritizes **low-latency model inference, offline operation, and robustness during continuous audio streaming**.

## Requirements

- Python 3.10+ recommended
- Working system audio input/output
- Piper binary available at:
  - Linux/macOS venv: .venv/bin/piper
  - (Adjust in code if your layout differs)

## Performance Notes

Typical end-to-end latency is approximately 1–2 seconds per spoken phrase on consumer CPU
hardware, depending on model size and audio conditions. Latency is managed through adaptive 
buffering and silence-based segmentation to balance responsiveness with transcription accuracy.

## Project Structure

src/ contains the real-time inference pipeline and orchestration logic.
models/ contains local ASR and TTS model files (not included in repo).
The system is modularized around ASR, NMT, and TTS boundaries for clarity
and extensibility.


Dedicated to my Kazakh grandparents — built so we have an easier time talking without them relying on 3rd party software. 
Still a work in progress, feel free to make improvements.
