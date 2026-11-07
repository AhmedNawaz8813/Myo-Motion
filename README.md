# MyoMotion (Still in the Development Phase)

Final Year Project 


MyoMotion is an advanced biomechanical coaching platform that fuses computer vision and electromyography by deploying YOLO, MediaPipe pose tracking, and spine aware feature extraction on a standard 2D camera feed, utilising sEMG calibrated deep learning architectures, to perform high precision, phase segmented muscle activation estimation (%MVC) and deliver real-time causal corrective feedback.



**Team:** Ahmed Nawaz · Aayan Sethi · Wajid Ali  


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

