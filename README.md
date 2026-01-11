# Realtime RU ↔ EN Speech Translator (Vosk + Argos Translate + Piper TTS)

A realtime speech tool that:
- **Always listens for Russian** (hands-free) and prints **RU → EN** when you stop speaking.
- Supports **English push-to-talk (SPACE)**: speak English, release SPACE, it translates **EN → RU** and **speaks Russian** using **Piper TTS**.



## Features

- Offline speech recognition with **Vosk**
- Offline machine translation with **Argos Translate**
- Offline TTS with **Piper**
- Push-to-talk English mode (SPACE)
- Hands-free Russian mode
- Built-in mic-mute during TTS playback to reduce self-listening feedback



## Project layout

Expected folder structure:

project/
main.py
Models/
vosk-model-small-ru-0.22/
...
vosk-model-small-en-us-0.15/
...
piper/
ru_RU/irina/medium/
ru_RU-irina-medium.onnx
ru_RU-irina-medium.onnx.json

> The script expects these exact default paths:
- Models/vosk-model-small-ru-0.22
- Models/vosk-model-small-en-us-0.15
- Models/piper/ru_RU/irina/medium/ru_RU-irina-medium.onnx
- plus sibling config: ru_RU-irina-medium.onnx.json



## Requirements

- Python 3.10+ recommended
- Working system audio input/output
- Piper binary available at:
  - Linux/macOS venv: .venv/bin/piper
  - (Adjust in code if your layout differs)

Python deps are installed via `requirements.txt` (below).



## Install

Create/activate a venv, then:

```
pip install -r requirements.txt
```
