# AI Desktop Assistant

A local-first, multimodal desktop assistant controlled by voice and computer-vision hand gestures. It automates common desktop tasks—creating and managing files, launching applications, browsing websites, and controlling active windows—running locally without cloud API dependencies.

The assistant also includes a few-shot **teachable gesture engine** to record and classify custom hand poses on-device via webcam in seconds using k-NN classification.

---

## Features

- **Continuous Voice Control**: Hands-free voice interaction using Voice Activity Detection (VAD) and local speech-to-text (`faster-whisper`). Starts listening on launch without requiring push-to-talk.
- **Offline Speech Synthesis**: Local text-to-speech with `pyttsx3`, synchronized across threads with an immediate **Esc** hotkey interrupt.
- **Fast Intent Routing**: Regex-based intent parsing for common tasks, with optional fallback to a local Ollama LLM for ambiguous phrasing.
- **Vision Hand Gestures**:
  - **Fist-Hold Window Close**: Hold a closed fist to target the focused window, with mandatory spoken yes/no confirmation before closing to prevent accidental data loss.
  - **Swipe Navigation**: Natural palm and finger swipes for page scrolling and browser tab switching, stabilized with aspect-ratio scaling and exponential smoothing.
  - **App-Aware Profiles**: Swipes adapt to the active application (e.g., remapping swipes to arrow keys in browser games).
- **Teachable Gestures**: Record a small set of samples (~20–30) through an interactive CLI/camera wizard, featurize them into rotation- and scale-invariant vectors, and bind them to custom hotkeys or actions.
- **Safe File Operations**: Trash-first deletions (`send2trash`), path traversal sanitization, conflict avoidance with automatic versioning `(1), (2)`, and starter boilerplate for common file extensions.
- **Cross-Platform OS Abstraction**: Window management adapter supporting Windows (via `pygetwindow`) and Linux X11 (via `xdotool`), with graceful degradation on unsupported platforms.

---

## Architecture

The project maintains a strict separation of concerns between intent classification ("decide") and execution ("do"). Intent handlers produce an immutable, validated `Intent` contract, and the executor applies safety checks and confirmation gates before running destructive actions.

```
                      ┌───────────────────────────┐
   Microphone ───────►│  Speech-to-Text (STT)     │ voice/stt.py (faster-whisper)
                      │  Continuous VAD listening │
                      └─────────────┬─────────────┘
                                    │ text
                      ┌─────────────▼─────────────┐
                      │      Intent Router        │ intent/rules.py (regex)
                      │   (Rules -> Local LLM)    │ intent/llm_fallback.py (Ollama)
                      └─────────────┬─────────────┘
                                    │ validated Intent
                      ┌─────────────▼─────────────┐
                      │    Executor / Dispatch    │ executor/dispatch.py
                      │ (Confirmation & Safety)   │
                      └───┬───────┬─────────┬─────┘
                          │       │         │
                 Files ───┘  Apps │         └── Browser
                                  ▼
                    executor/files.py, apps.py, browser.py

                      ┌───────────────────────────┐
   Webcam ───────────►│   Hand Tracker & Vision   │ gestures/tracker.py (MediaPipe)
                      └─────────────┬─────────────┘
                                    │ landmarks
                      ┌─────────────▼─────────────┐
                      │    Gesture Recognizer     │ gestures/recognizer.py (swipes/fist)
                      │   & Teachable Store       │ gestures/teachable.py (k-NN)
                      └─────────────┬─────────────┘
                                    │
                      ┌─────────────▼─────────────┐
                      │    Platform OS Adapter    │ os_adapter/ (Windows / Linux X11)
                      │  (Active Window & Close)  │ context/active_window.py
                      └───────────────────────────┘
```

---

## Installation

### Prerequisites
- **Python**: 3.10 – 3.13
- **Operating System**: Windows 10/11 or Linux (X11 with `xdotool`)
- **Hardware**: Microphone (for voice mode) and webcam (for gesture mode)

### Setup

```bash
git clone <repository-url>
cd AIDesktopAssistant

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Optional)* For LLM fallback on ambiguous phrasing, install [Ollama](https://ollama.com) and pull the model:
```bash
ollama pull qwen2.5:1.5b
```

---

## Usage

The application provides multiple execution modes:

### 1. Text Mode (No microphone or camera required)
```bash
python assistant.py
```
Type commands directly into the terminal:
```text
Assistant is listening (text mode). Type a command, or 'quit' to exit.

> create a python file called calculator
Assistant: Created calculator.py in Desktop
> delete calculator.py
Assistant: Are you sure you want to delete 'calculator.py'?
(y/n) > y
Assistant: Moved calculator.py to trash
> open chrome
Assistant: Opening Chrome
```

### 2. Voice Mode
```bash
python assistant.py --voice
```
- Listens continuously via VAD.
- Press **Esc** at any time to interrupt speech output or listening.
- Say *"quit"*, *"exit"*, or *"stop"* to end the session.

### 3. Gesture Mode
```bash
python assistant.py --gesture
```
- Swipe up/down to scroll, or swipe left/right to switch browser tabs.
- Hold a closed fist for ~0.7 seconds: the assistant prompts for spoken yes/no confirmation before closing the focused window.
- Press **q** in the camera window to exit.

### 4. Dual Mode (Simultaneous Voice & Gestures)
```bash
python assistant.py --all
```
Runs gesture recognition and continuous voice interaction concurrently with synchronized microphone access.

### 5. Custom Teachable Gestures
```bash
# Record and train a new gesture
python assistant.py --teach-gesture

# View all trained custom gestures
python assistant.py --list-gestures
```

---

## Platform Support

| Platform | Active Window Detection | Window Closing | Details |
|---|---|---|---|
| **Windows** | Supported | Supported | Backend: `pygetwindow` |
| **Linux (X11)** | Supported | Supported | Backend: `xdotool` (`sudo apt install xdotool`) |
| **Linux (Wayland)** | Fallback | Fallback | Falls back to `NoOpAdapter` (asks user for target app) |
| **macOS** | Fallback | Fallback | Falls back to `NoOpAdapter` |

---

## Running Tests

All modules include unit and regression tests using synthetic inputs, mocks, and temporary directories. The test suite does not modify personal files or require connected hardware.

```bash
pytest tests/ -v
```

---

## Configuration

Settings and mappings can be modified without altering source code:
- **`config/settings.yaml`**: Audio parameters, gesture confidence thresholds, camera resolution, and default directories.
- **`config/apps.yaml`**: Supported applications, aliases, launch commands, and process names.
- **`config/gesture_profiles.yaml`**: App-scoped gesture action overrides (e.g., arrow keys for browser games).

---

## License

This project is licensed under the MIT License.
