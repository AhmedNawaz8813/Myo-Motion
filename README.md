# MyoMotion

**BSCS Final Year Project — FAST-NUCES Peshawar**

MyoMotion coaches users through exercises using only a phone camera. It checks posture, estimates per-muscle activation (%MVC) in each rep phase, and delivers corrective cues in real time.

**Team:** Ahmed Nawaz · Aayan Sethi · Wajid Ali  
**Supervisor:** Dr. Muhammad Amin  
**Stage:** FYP-1 (mid-defense prep)

---

## Project Structure

```
MyoMotion/
├── scripts/                   # Python scripts
│   └── deadlift_tracker.py    # MediaPipe-based deadlift angle tracker
├── models/                    # Large binary model files (not committed to git)
│   └── pose_landmarker_heavy.task
├── docs/                      # Project context and documentation
│   └── Context.md
├── CLAUDE.md                  # Agent context file
├── requirements.txt           # Python dependencies
└── .gitignore
```

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the MediaPipe model (if not present)
#    Place pose_landmarker_heavy.task into models/
```

## Running

```bash
# Always run from the project root
python scripts/deadlift_tracker.py
```

Press **Q** to quit the live window.

## Tech Stack

| Layer | Tool |
|---|---|
| Pose estimation | MediaPipe Pose Landmarker (Tasks API) |
| Computer vision | OpenCV |
| EMG hardware | Arduino Uno + 3× AD8226 modules |
| ML target | %MVC regression (scikit-learn / PyTorch) |

