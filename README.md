# Face Mario

A 2D platform game controlled by your face and hand gestures, with keyboard
controls as a fallback. The player's head uses a live, face-shaped webcam
cutout. The included pixel-art background is loaded from `background.png`.

## Controls

### Webcam

- Tilt your head left or right to move.
- Open your mouth to jump; keeping it open longer makes the jump higher.
- Make a fist to punch.

Keep your head straight and your mouth closed during the short startup
calibration.

### Keyboard

- **Left/Right** or **A/D**: move
- **Space**, **W**, or **Up**: jump
- **F** or **J**: punch
- **C**: recalibrate face controls
- **P**: pause
- **R**: restart
- **Esc**: quit

## Run on Windows

1. Install Python 3.11 and ensure the Python Launcher (`py`) is available.
2. Double-click `run_windows.bat`. It creates a virtual environment and
   installs the packages in `requirements.txt`.
3. Allow camera access if Windows asks.

The first run downloads the MediaPipe face and hand tracking models, so an
internet connection is required. To start the game manually from the project
folder:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

If your camera is not detected, check that other applications are not using
it. Keyboard controls work without a camera.

## Project files

- `main.py` starts the game.
- `hand_ar.py` contains the game, webcam tracking, and controls.
- `background.png` is the game background.
- `requirements.txt` lists the Python dependencies.
- `run_windows.bat` sets up and launches the game on Windows.
