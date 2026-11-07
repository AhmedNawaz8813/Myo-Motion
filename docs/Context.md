# MyoMotion — Full Project Context

_Last updated: 2026-10-12. Self-contained: a reader with no other material should understand the project, its documents, decisions and open problems._

Sources merged here:
- the team's description of the project
- the proposal defense deck (14 slides)
- the original UML diagrams
- the SRS (6 pages)
- the FYP-1 mid-defense rubric
- the sensor hardware details
- the team's **final literature review deck with speaker notes** (`MyoMotion_Literature_Review.pptx`)
- the team's **final UI design board** ("MyoMotion – Final UI Design": 9 phone screens, see §13)
- all design and review work done with Claude

---

## 1. At a glance

| | |
|---|---|
| **Project name** | **MyoMotion** (current, used in the SRS, the literature deck and the latest diagrams). "AIFIT"/"AI-FIT" was an interim name in some older files; don't use it. |
| **SRS title** | MyoMotion: Vision-Extracted Biomechanics and sEMG-Estimated Muscle Recruitment System |
| **Deck title** | Exercise Analysis Using Computer Vision and sEMG-Based Muscle Activation Prediction |
| **Programme** | BSCS Final Year Project (FYP-1 + FYP-2), FAST-NUCES **Peshawar campus** |
| **Supervisor** | Dr. Muhammad Amin |
| **Team** | Ahmed Nawaz (23P-0550), Aayan Sethi (23P-0503), Wajid Ali (23P-0606) |
| **What it is** | An app that coaches a user during exercise from camera video: checks posture, estimates target-muscle activation (%MVC) per rep phase, and gives corrective cues |
| **How it is used** | Camera only. The user films with one phone; no sensors on the user. sEMG is used by the team to build the training dataset |
| **Video input** | **Live** coaching (≤ 33 ms per frame) and **uploaded** video (offline batch) |
| **Exercises** | **Many**, together covering muscles across the whole body (upper body, lower body, trunk). The exact list is **not final**; it will be chosen after more research (§2). The squat is the worked example in the wireframes, diagrams and literature deck |
| **Key tech** | YOLOv5 (person detection) + MediaPipe Pose + spine module (SpinePose or trigonometric); 3 × AD8226 sEMG modules on an Arduino Uno; ML/DL activation model |
| **Stage** | FYP-1; preparing for the **mid defense** (rubric in §11). Literature review done. Final UI wireframes done (9 screens, §13). |

---

## 2. The idea

### What it does
During each supported exercise, MyoMotion:
1. checks posture
2. estimates how much each target muscle works, overall and per rep phase
3. tells the user how to adjust, so form is correct and target-muscle activation is maximised

### How it works for the user
- The user props a phone 2–3 m away and films the set from the view the app asks for that exercise (side or front). No sensors, no setup.
- A model trained on the team's **synchronised video + sEMG dataset** estimates muscle activation from the video alone.
- The app shows the skeleton, rep phases, estimated activation per muscle and corrective cues, live or after uploading a video.

### Exercise scope (not final)
- MyoMotion targets **many exercises** so that, together, they work muscles **across the whole body**: upper body, lower body and trunk. It is **not** a squat-only app.
- The team has kept the list **generic for now** and will pick the exact exercises after more research.
- The **squat** is only the worked example used so far: in the UI wireframes, the diagrams and most of the literature evidence.
- **Suggested criteria for choosing exercises** (Claude's suggestion, not yet agreed by the team):
  - together they cover the main muscle groups of the upper body, lower body and trunk
  - each can be seen clearly by one phone camera from one view (side or front)
  - repetition based, with clear down / hold / up phases
  - target muscles can be reached with surface electrodes (SENIAM sites)
  - safe with bodyweight or light load
  - known posture variations that change which muscle works, so the cues matter
- **What the list changes:**
  - each exercise needs its own recorded video + sEMG data (MuscleMap: 97.8% → 64.2% mAP on unseen activities)
  - with 3 sensors, each exercise needs its own muscle set, so more exercises means more recording blocks per participant
  - each exercise needs its own form rules (angle limits, cues) and camera view
  - adding exercises in stages, with one layer retrained (Lehmler), keeps the workload manageable

### Research gap (use this framing)
Video-to-muscle estimation already exists: *Muscles in Action* (ICCV 2023) and *OpenCap* (PLOS Comp Biol 2023). The gap is that **no system combines**:
- spine-aware single-camera activation estimation
- reporting per rep phase
- posture cues tied to muscle activation (not just joint angles)
- live coaching on a phone

Gap-slide wording: "No existing system covers all seven features; MyoMotion is the only one that targets every column."

**Research question:** can spine-aware features from one camera estimate target-muscle activation per rep phase well enough to coach posture?

### Core argument (from the speaker notes)
Small posture changes shift which muscle does the work and by how much (gluteus maximus +13–61% with a wider stance, +65% from shallow to thigh-parallel), so checking joint angles alone can't tell which muscles are working. This is why MyoMotion estimates activation from posture features and ties its cues to the target muscle.

---

## 3. How it works

### 3.1 Phases (SRS)
1. Research and dataset collection (synchronised video + sEMG; train and evaluate)
2. End-user app (camera only)

### 3.2 Training pipeline
1. Configure the experiment, then place electrodes (SENIAM).
2. Check the signal-to-noise ratio; reposition if low. Then record MVC per muscle.
3. Trigger the sync cue (LED/beep), then record video and sEMG in parallel, timestamped.
4. **sEMG:** the AD8226 already outputs a rectified, smoothed envelope. Software normalises it to %MVC, giving **Y(t)**.
5. **Video:**
   - YOLOv5 person box → MediaPipe 33 landmarks
   - spine estimation (SpinePose 9 spine points, or the trig estimate)
   - joint angles → features **X(t)**
6. Align by timestamp → dataset → validate.
7. Train → evaluate with **leave-one-subject-out (LOSO)**, which is stricter than Muscles in Action's same-subject testing.

### 3.3 Deployment (per frame)
1. Pose → spine → features → rep phase → activation (%MVC per muscle) → form assessment against the exercise rules.
2. Corrective cue and overlay. If no person is in frame, warn after 1 s.
3. After each rep: compensation detection.
4. At the end: report (form score, activation efficiency, compensations); export to PDF or annotated video.

### 3.4 Data-collection protocol (posture variations)
- Record each participant in several posture variations of each exercise (squat example: stance width × foot angle × depth) so the model sees how posture shifts activation.
- Keep **load and tempo the same**, **randomise the order**, rest between variations, and record **several reps** per variation.
- Label which variation gives the best target-muscle share vs compensating muscles; this defines what the cues coach toward.

### 3.5 Spine module (decision rule in the speaker notes)
MediaPipe has **no spine points**; its only trunk points are the shoulders (11, 12) and hips (23, 24).

- **SpinePose** (Khan et al., CVPRW 2025):
  - adds 9 spine keypoints (cervical → sacral); lightweight; public code
  - 2D only, so **film from the side** to see lumbar rounding (the camera view is chosen per exercise)
- **Trigonometric route:** shoulder/hip proxies give **trunk lean** and catch obvious back faults (Ko et al. flag a non-neutral spine with MediaPipe landmarks only). But OpenCap found that hip and shoulder keypoints alone gave **12.1–39.2° errors** in lumbar extension, pelvic tilt and hip flexion, compared with their marker-augmented model. And the same trunk lean can come from a safe hip hinge or a rounded back (Straub & Powers).
- **Rule:** if SpinePose can give **real-time feedback on a phone**, use it. Otherwise use the trigonometric route (trunk lean only).
- **Ground truth for spine accuracy:** spinous-process markers (C7, T12, L3, S1) filmed side-on. Spinous markers are more stable than lateral ones.
- **SRS target:** lumbar angle MAE ≤ 5°.

### 3.6 Model ideas (from the speaker notes)
- **Starting point:** Ko et al.'s public YOLOv5 + MediaPipe + classifier pipeline (94.7 to 98.9% posture accuracy across three lifts is a realistic target for posture checking).
- **Attention layer** (from Chariar): weight the important frames, like the bottom of a squat, peak activation or likely back rounding. In live mode it must be **causal** (past frames only).
- **Self-supervised pretraining** on unlabelled exercise videos to cut the labelling needed (from Parmar / Fitness-AQA).
- **Cue style** (from AIFit): name the joint, direction and degrees.
- Predict **%MVC strength** against real sEMG, not just active/inactive (MuscleMap's weakness). **Every exercise needs its own training data** (MuscleMap drops from 97.8% to 64.2% mAP on new activities), so add exercises in stages.
- **Candidate models:** RF/SVM/LSTM/CNN/TCN (+ attention). Regression X(t) → Y(t). LOSO evaluation.

---

## 4. Hardware (decided)

### 4.1 Parts
- Arduino Uno R3
- **3 × AD8226 EMG modules** (board marking "ECGP2-Q100")
- 3.5 mm 3-lead cables
- 3M snap gel electrodes
- 9V batteries (HW Hi-Watt 6F22)
- jumpers and a USB A-B cable
- laptop
- one 30+ FPS camera

### 4.2 AD8226 facts
- **Output:** a rectified, smoothed **envelope**, not raw EMG.
  - No software filtering or rectifying; just normalise to MVC.
  - Fatigue can use **amplitude signs** (RMS/MAV rise) but **not the frequency drop**.
  - Sample at 500 Hz–1 kHz per channel.
- **Power:** ±9V from two 9V batteries. All three modules can share one pair through a common ground.
- **Gain:** set by a potentiometer; set it once and mark it.
- **Electrodes:** 3 per module (2 on the muscle about 2 cm apart along the fibres + 1 bony reference), so 9 per recording.
- **Wiring:** outputs → A0, A1, A2.

### 4.3 Session rules
- fresh batteries; **redo MVC every session**
- **Python serial logger** (not the Serial Plotter), binary at ≥ 250,000 baud, timer-interrupt sampling
- LED flash in camera view for sync
- **laptop on battery or a USB isolator** while electrodes are on
- shave, abrade, wipe with alcohol, tape the cables

### 4.4 Three-sensor plan
- **General rule:** 3 muscles per recording block, one side. For each exercise, pick the target muscle, a posture-sensitive muscle and a likely compensator. Covering muscles across the whole body means several blocks per participant, with electrodes moved and MVC recorded for each muscle.
- **Worked example, squat block** (the team's notes list vastus lateralis "on our muscle list"), right side:

| Sensor | Muscle | Role | SENIAM site | Reference |
|---|---|---|---|---|
| 1 | Gluteus maximus | Main posture-sensitive target | 50% sacrum → greater trochanter | Iliac crest |
| 2 | Vastus lateralis | Prime mover; cleanest signal | 2/3 ASIS → lateral patella | Kneecap |
| 3 | Lumbar erector spinae (longissimus) | Compensation; links to spine | 2 finger-widths lateral to L1 | Iliac crest |

- **Optional Block B:** biceps femoris + gastrocnemius + vastus lateralis as an anchor.
- Cross-muscle ratios only within a block.
- **Bilateral asymmetry is not measurable** without a dedicated left/right block.

---

## 5. SRS (6 pages, LaTeX) — summary and known defects

### 5.1 User classes, environment and constraints
- **User classes:** the SRS lists Researcher (electrodes, MVC, sync, validation, training) and Athlete (live/upload). **The team has since settled that the app has only one actor, the User (Athlete)**; the team's own data collection is Phase 1 work, not a user class (see §5.3 item 2).
- **Environment:**
  - GPU for training
  - containerised modular deployment
  - mobile or web app with cloud inference
  - one 30+ FPS camera
- **Constraints:**
  - serial bandwidth
  - no MediaPipe spine landmarks (use proxies and trunk crops)
  - safety during load and MVC
- **Assumptions:** good lighting, full body visible, users follow the protocol.
- **Disclaimer:** **not a medical device.**

### 5.2 Requirements
**Functional requirements:**
- FR-DAQ-001: baseline / SNR
- FR-DAQ-002: MVC
- FR-DAQ-003: sync
- FR-CV-001: frame validation
- FR-CV-002: modular spine
- FR-CV-003: joint angles
- FR-CV-004: X(t)
- FR-REP-002: form validation
- FR-ML-001: X→Y regression
- FR-ML-002: compensation
- FR-ML-003: activation efficiency
- FR-UI-001: overlay
- FR-UI-002: cues
- FR-UI-003: PDF export

**Non-functional requirements:**
- batch mode for uploads; live ≤ 33 ms per frame
- recover from frame drops, noise and disconnects
- legible from 2 m; guided sEMG setup (electrodes, MVC) (SRS wording says "for the Researcher"; in v2 move this to the Phase 1 data collection section, since the app has no Researcher)
- swappable pose backend, spine model and exercise catalogue
- anonymisation and encryption

**Acceptance criteria:**
- sync drift ≤ 33 ms
- lumbar MAE ≤ 5°
- phase boundaries ±150 ms
- LOSO r ≥ 0.80 for prime movers
- asymmetry warning > 15%
- latency ≤ 33 ms
- sensor-drop recovery within 2 s

### 5.3 Defects to fix in v2
1. **Typesetting:** "M yoM otion" (maths mode in LaTeX); use `\textit{}`.
2. **User classes:** drop Researcher as a user class. The only user is the User (Athlete); describe the team's data collection under Phase 1 instead.
3. **Asymmetry criterion** conflicts with 3 one-side sensors.
4. **Cloud inference vs ≤ 33 ms:** run pose on the device.
5. **IDs:**
   - FR-REP-001 is used but never defined (no phase-segmentation requirement)
   - FR-ML-002 has two meanings
   - FR-ML-001 lost its arrow
6. **Missing requirements:** login, password reset, profile, exercise selection, history, storage, upload processing. Describe the team's dataset and training work under Phase 1, not as app requirements.
7. **Vague:**
   - SNR threshold
   - "a technique, but not limited to"
   - activation efficiency formula undefined
   - lumbar "baseline" undefined
   - "anonymised" video impossible (blur faces or store landmarks only)
8. **r ≥ 0.80** is ambitious; use a tiered target.
9. **Hardware list:** quantities; Python logger; battery-safety rule.
10. **Y(t) vs Ŷ(t):** measured vs predicted.
11. **Wording:** "shall" consistently.
12. **Missing sections:** external interfaces, traceability, references.
13. **Typos:** muslces, calcualte, specifi, Electromygraphic, trignometric, Mediapipes.

---

## 6. UML diagrams

### 6.1 Current diagram set (version 3, 2026-10-12, folder `diagrams/`)
Replaces the partner's Software Ideas Modeler (SIM) diagrams and the older root-level `MyoMotion_Class_Diagram.*` / `MyoMotion_Sequence_Diagram.*`.

**Team rules for the diagrams (settled 2026-10-12):**
- **Only one actor: the User (Athlete).** There is no Researcher actor, role or class anywhere. The team does collect the sEMG dataset, but that is not a feature of the app.
- **The use case and deployment diagrams show only what the released app does.** Nothing from building the app (data collection, training) goes in them.
- Use case layout: the primary actor (User) on the left; secondary actors (external systems the app calls) on the right. The MyoMotion server is part of the system, not an actor.

**Shared style:**
- **exercise-generic** (no squat-only labels)
- plain black/grey/white (`mono_style.iuml`)
- every diagram as `.puml`, `.png` and `.svg`
- `MyoMotion_Diagrams.pdf` has all 8 diagrams, one per page
- `README.md` lists the files, the changes, and what to fix after opening in SIM

**The 8 diagrams:**
1. **Use case (released app):** one primary actor, User (Athlete), on the left. Solid association lines.
   - User: Create Account, Log In, Reset Password, Manage Profile, Start Live Coaching, Upload Exercise Video, View Analysis Report, View Session History.
   - Live «include»s Capture Live Video, Check Body in Frame, Give Live Cue, Choose Exercise and Target Muscle, and Analyse Exercise Set. Upload «include»s Choose Exercise, Pick Video from Gallery, Validate Video and Analyse Exercise Set. Reset Password «include»s Send Reset Link by Email.
   - Analyse Exercise Set «include»s Track Pose and Spine, Detect Rep Phases, Estimate Muscle Activation, and Assess Form / Detect Compensation.
   - Replay a Rep ("Show me") and Export Report «extend» View Analysis Report.
   - Secondary actors on the right («system»): Device Camera (Capture Live Video), Text to Speech (Give Live Cue), Phone Gallery (Pick Video from Gallery), Email Service (Send Reset Link by Email).
2. **Activity, upload:**
   - choose exercise and video → validate (retry loop) → load model and form rules for the exercise
   - per-frame loop: YOLO → [person found?] → MediaPipe → spine → X(t)
   - [enough valid frames?] → reps/phases → %MVC → form → compensation → report → [export?] → [analyse another?]
3. **Activity, live:** [camera permission?] → choose exercise and muscle → framing loop → Ready → record → per-frame loop with:
   - [person found?]
   - [form issue past the exercise's limit → spoken cue]
   - [rep completed?]
   - [no person > 1 s?]
   - [Stop?] → report.
4. **Sequence, live coaching:** User, CoachingScreen, SessionController, Database, CameraStream, PersonDetector (YOLO), PoseEstimator, SpineEstimator, FeatureExtractor, PhaseDetector, ActivationModel, FormAnalyzer, CompensationDetector.
5. **Class diagram:** one package (MyoMotion App), polyline routing, no FKs, no methods.
   - User (no role), Exercise (body_region, camera_view, form_rules), Muscle (body_region), WorkoutSession, ActivationModel, FrameFeatures, Repetition (range_of_motion_deg), PhaseSegment, MuscleActivation, AnalysisReport, FormIssue; enums RepPhase, InputType.
6. **ERD:** crow's foot; 13 app tables plus 6 training dataset tables in a separate group.
   - App tables include `exercises.body_region/form_rules` and `muscles.body_region`; `users` has no role column.
   - Training dataset (built by the team, not used by the app at run time): participants, recording_sessions (no link to users), session_muscles (channel 1 to 3), mvc_records, recordings (variation JSON), model_training_data.
7. **Architecture:** hardware (3 muscles per block chosen per exercise; camera view per exercise; LED driven by the Arduino) → offline training pipeline → app pipeline (activation model per exercise, form analyser with exercise form rules) → user outputs. This is the only diagram that shows the sEMG data collection setup.
8. **Deployment (released app):**
   - phone app with on-device inference (activation models per exercise or shared)
   - FastAPI server with upload worker, export and model registry; PostgreSQL; object storage
   - no data collection laptop, Arduino or sensors (those are only used while building the model)
   - **Assumption:** a mobile app.

**Dropped in version 3:** the Researcher actor and its use cases, the data collection activity (swimlanes), the recording sequence, the class diagram's research package and `UserRole`, and the deployment's research laptop setup. If the report ever needs a data collection diagram, draw it as a Phase 1 process without a Researcher actor.

**SIM files (`diagrams/sim/*.simp` for diagrams 1 to 7, plus `MyoMotion_All_Diagrams.simp`):**
- **How they were made:** a vendored, reviewed copy of the unofficial `simp-file-builder` 0.1.0 (MIT; SIM 15.21 format). Generator in the scratchpad: `tools/make_simp.py`, `simp_builder.py`, `check_simp.py`.
- **Checked:** every file parses and has 0 broken internal references.
- **Limits:**
  - no multiplicity or composition fields (written as `[1 : 0..*]` and "(composition)" in the labels)
  - no `alt` (pairs of `opt`)
  - no enumerations
  - no actor stereotypes (add «system» to the right side actors by hand if wanted)
  - **no deployment diagram**
- **Not yet confirmed:** that they open in SIM.

### 6.2 Detailed design set (older, titled "AIFIT")
Folder `aifit/`: a detailed class diagram (~40 classes, 5 packages), a 45-step sequence diagram and a swimlane diagram.

**Outdated:** superseded by the MyoMotion set above. Don't reuse them; redraw from the current design if the report needs more detail.

### 6.3 Original team diagrams — known issues (fixed or dropped in the 2026-10-12 set)
- **User activity:** loop back on invalid video, guards, "enough valid frames?", redundant display step.
- **Use case:**
  - solid actor lines
  - extend base use case
  - data collection steps (MVC, sensor setup) removed: the use case shows only the released app
  - add a compensation use case
- **Architecture:**
  - Form Analyzer from features
  - split phase/compensation
  - route outputs through feedback
  - sync cue
  - MCV → MVC
  - remove the watermark
- **Data collection activity:** dropped (not part of the released app).
- **Duplicate user activity diagram:** drop it.

---

## 7. Literature review

### 7.1 Team's deck: `MyoMotion_Literature_Review.pptx` (11 slides, university template; updated 2026-10-12)
- **Slides 1–5 "Literature Review (n of 5)":** tables with Sr No | Title | Year | Key Findings | Strengths | Limitations | **Relevance to MyoMotion**. 11 papers, 2 per slide (3 on slide 5). Each paper has a detailed **speaker note**.
- **Slide 6 "Comparison with Existing Work":** gap matrix (8 systems + MyoMotion × 7 features).
- **Slide 7 "Research Gap and Our Contribution":** "The Gap:" (4 lines) and "Our Contribution:" (3 lines).
- **Slide 8 "Problem Statement":** just the one-sentence problem statement, centred on the slide (the team removed the supporting lines; those facts are still in the speaker notes).
- **Slide 9 "MyoMotion's Solution":** intro line, bold-label lines (Pose Tracking: **YOLO + MediaPipe** + spine module; Muscle Estimation; Live Coaching), bold closing line.
- **Style rule from the team:** plain, "made by a human" look: black/grey/white only, plain text with bold labels in the template's serif font (Century Schoolbook), no cards, icons, arrows or coloured callouts. Say "YOLO" (generic) for MyoMotion's own pipeline; "YOLOv5" only when describing Ko et al. No hyphenated words or dash number ranges in slide text or notes (write "real time", "12 to 39°"); plain student wording. Official paper titles and IEEE references stay exactly as published. Matrix row [11] is labelled "Parmar et al., 2022 [11]".
- **Slides 10–11:** IEEE references [1]–[11].
- Every slide has speaker notes (references excepted). Builder script: scratchpad `tools/build_lit.py` + `tools/lit_content.py`.
- Hsiao et al. (EMG wearables) was dropped compared with Claude's earlier draft.

| # | Paper | Key findings in the deck | Relevance to MyoMotion |
|---|---|---|---|
| 1 | Ko et al., IEEE Access 2024 | YOLOv5 + MediaPipe + RF; posture accuracy squat 98.9%, bench 96.1%, deadlift 94.7%; real time; code public | **Pipeline the team plans to use**; realistic accuracy target; doesn't estimate muscles |
| 2 | Bawa & Banitsas, Sensors 2022 | Arduino Uno + MyoWare vs Delsys; ICC 0.74–0.92; fatigue via RMS/MAV up, mean frequency down; $150 vs $12,000; 7 people, one muscle (vastus lateralis), one channel | Supports low-cost hardware; AD8226 envelope allows amplitude fatigue signs, not frequency |
| 3 | Straub & Powers, IJSPT 2024 | Posture shifts work between hips/knees/lower back; glute max +13–61% (wide stance), +65% (shallow → parallel); quads/hamstrings little change | **Main argument**: posture changes activation, so estimate muscles, not just angles; justifies the spine module (hip hinge vs rounded back); glutes as the posture-sensitive target |
| 4 | Khan et al. (SpinePose), CVPRW 2025 | SpineTrack ~51k images / 58k people, 9 spine points; AP ~89–91 vs 0 for models without spine points | **Spine decision rule**; film side-on; 2D only |
| 5 | Lehmler et al., J Neural Eng 2022 | Fine-tuning beats person-only models by ~12 points and the unchanged pretrained model by 5; 22% fewer epochs; first or last layer is enough | **Transfer learning** for sEMG models: pretrain, then fine-tune one layer on a small dataset (useful for the team's small dataset and for adding exercises) |
| 6 | Chariar et al., IEEE Access 2023 | MediaPipe + Bi-GRU + attention; 94% over 7 squat classes; cues adapted to body proportions | **Attention layer** idea (causal in live mode) |
| 7 | Muscles in Action, ICCV 2023 | 12.5 h, 10 people, 15 exercises, 8 muscles; RMSE 8.0–20.4 vs 10.5–25.7; generates corrected motions | Proves the core idea; team uses stricter LOSO testing |
| 8 | OpenCap, PLOS Comp Biol 2023 | 2 iPhones + cloud; squat asymmetry AUC 0.83 lab / 0.90 field (n = 100); **2 min kinematics, 35 min muscle forces per squat**; hip/shoulder keypoints alone → 12–39° trunk/pelvis/hip errors | Video muscle estimates work in squats; supports the spine module; not real time |
| 9 | MuscleMap, ACM MM 2024 | 15K+ clips, 135 activities, 20 groups; 97.8% → 64.2% mAP on new activities; active/inactive only | Justifies %MVC vs sEMG; shows each exercise needs its own training data |
| 10 | AIFit, CVPR 2021 | 3D pose → rep segmentation (IoU 0.73) → compare with trainer → text feedback (~80% accurate); Fit3D 13 people, 37 exercises | Supports rep-phase detection; cue style (joint, direction, degrees) |
| 11 | Parmar et al. (Fitness-AQA), ECCV 2022 | Self-supervised on unlabelled gym video; small expert set; barbell height to match poses | Works in real gyms; **pretraining on unlabelled exercise videos** |

### 7.2 Gap matrix (deck slide 6)
Columns (7): Video, Spine detail, sEMG, Activation from video, Rep phases, Corrective feedback, Real time. Rows:
- Ko [1]
- SpinePose [4]
- Chariar [6]
- MIA [7]
- OpenCap [8]
- MuscleMap [9]
- AIFit [10] (real time "–")
- Fitness-AQA [11]
- MyoMotion (all ✓)

### 7.3 Review of the deck (fix before the defense)
1. **Verify these claims in the papers** (not confirmed by Claude):
   - Ko: "no GPU, CPU only, works on phones"; Ko flags a non-neutral spine
   - SpinePose: "first dataset and model for detailed spine tracking"
   - MIA: "tested on the same people it trained on"
   - Bawa: "noisier, lower %MVC"
   - Parmar: barbell-height matching
2. **OpenCap nuance:** the 12–39° error compares keypoint-only kinematics with OpenCap's marker-augmented model, not absolute ground truth. Phrase it as "hip and shoulder keypoints alone gave 12–39° larger errors".
3. **References:**
   - Straub pp. 490–501: **verified**.
   - Lehmler vol. 19, no. 5, art. 056039, doi 10.1088/1741-2552/ac9860: **not confirmed**; check on the IOP page.
   - Page numbers are still missing for [4], [7], [10].
4. **Wording and typos:** fixed in the 2026-10-12 update (Ko grammar, "Only 3 exercises", Straub's informal aside, SpinePose "AP 89–91, whereas", "person-only", Chariar strengths, AIFit "finally", "Fitness-AQA", "Does not track muscle activity").
5. **Scope wording (squat only):** the deck still reads as squat-only in places. Reword to the many-exercise scope:
   - MuscleMap relevance ("start with just one exercise, the squat") and its note
   - OpenCap relevance ("our first exercise")
   - Straub relevance ("glutes are our main target")
   - Problem Statement ("check squat posture")
   - Solution slide (sensors listed only for the glutes, vastus lateralis and lower back)
6. **Timing:** the notes run about 1–1.5 min per paper (about 12 min total). For a mid defense, prepare a 3–4 minute version: one sentence per paper plus the gap slide.
7. **Rendering:** hyperlinked titles break mid-word in LibreOffice previews only; they're fine in PowerPoint.

### 7.4 Deck references [1]–[11] (IEEE, as in the deck)
[1] Y.-M. Ko, A. Nasridinov, and S.-H. Park, "Real-time AI posture correction for powerlifting exercises using YOLOv5 and MediaPipe," IEEE Access, vol. 12, pp. 195830–195853, 2024, doi: 10.1109/ACCESS.2024.3506940.
[2] A. Bawa and K. Banitsas, "Design validation of a low-cost EMG sensor compared to a commercial-based system for measuring muscle activity and fatigue," Sensors, vol. 22, no. 15, art. 5799, 2022, doi: 10.3390/s22155799.
[3] R. K. Straub and C. M. Powers, "A biomechanical review of the squat exercise: Implications for clinical practice," Int. J. Sports Phys. Ther., vol. 19, no. 4, pp. 490–501, 2024, doi: 10.26603/001c.94600.
[4] M. S. U. Khan, S. Krauß, and D. Stricker, "Towards unconstrained 2D pose estimation of the human spine," in Proc. IEEE/CVF CVPRW, 2025, arXiv:2504.08110.
[5] S. J. Lehmler, M. Saif-ur-Rehman, T. Glasmachers, and I. Iossifidis, "Deep transfer learning compared to subject-specific models for sEMG decoders," J. Neural Eng., vol. 19, no. 5, art. 056039, 2022, doi: 10.1088/1741-2552/ac9860. _(volume/DOI unverified)_
[6] M. Chariar, S. Rao, A. Irani, S. Suresh, and C. S. Asha, "AI trainer: Autoencoder based approach for squat analysis and correction," IEEE Access, vol. 11, pp. 107135–107149, 2023, doi: 10.1109/ACCESS.2023.3316009.
[7] M. Chiquier and C. Vondrick, "Muscles in action," in Proc. IEEE/CVF ICCV, 2023.
[8] S. D. Uhlrich et al., "OpenCap: Human movement dynamics from smartphone videos," PLOS Comput. Biol., vol. 19, art. e1011462, 2023, doi: 10.1371/journal.pcbi.1011462.
[9] K. Peng et al., "Towards video-based activated muscle group estimation in the wild," in Proc. 32nd ACM Int. Conf. Multimedia, 2024, pp. 4495–4504, doi: 10.1145/3664647.3680630.
[10] M. Fieraru, M. Zanfir, S. C. Pirlea, V. Olaru, and C. Sminchisescu, "AIFit: Automatic 3D human-interpretable feedback models for fitness training," in Proc. IEEE/CVF CVPR, 2021.
[11] P. Parmar, A. Gharat, and H. Rhodin, "Domain knowledge-informed self-supervised representations for workout form assessment," in Proc. ECCV, 2022, doi: 10.1007/978-3-031-19839-7_7.

### 7.5 Extended material (Claude-made, older "AIFIT" naming)
- **Claude Docs "AIFIT Literature Review"** (https://claude.ai/artifact/A4BJy59y8MsDUkuCaFaeFQ): 26 references across 7 themes, Tables A/B, defense Q&A.
- **`AIFIT_Literature_Tracker.xlsx`:** 26 papers + 10 to-read.
- **Additional verified papers not in the deck:**
  - Kotte 2023 (rule-based, 5-user qualitative)
  - BlazePose 2020 (33 keypoints, > 30 fps on a phone)
  - MediaPipe 2019
  - Xin 2022 (skin-marker lumbar error 13.5 ± 6.5 mm)
  - SENIAM / Hermens 2000
  - Burden 2010 (isometric MVC normalisation)
  - Avila 2023
  - Türker 2013
  - García-Valverde 2025 (no significant squat-variant differences)
  - Buonsenso 2025 (lat pulldown)
  - RepNet / Dwibedi 2020 (rep counting)
  - Hsiao 2020 (IMU + EMG bench coach, 91.84%)
  - Zhou 2020
  - Bengtsson 2018, Keogh 2017 (injuries)
- **Old deck problems:** reference [4] Beyki et al. 2026 could not be found; the "modified MediaPipe" figure came from an infant-height paper; slides cited web links.

---

## 8. Proposal defense deck (14 slides) — fixes
- typos: Jont, recruitement, acvtivation
- the "Objectives" slide is the Gantt chart; add objectives and retitle the Gantt slide "Timeline"
- define α (SRS: r ≥ 0.80)
- remove the empty team-table row
- reframe the gap
- add live coaching and video upload
- **Expected Result** mock-up shows 5 muscles and a left/right claim → change to 3 muscles, no asymmetry

---

## 9. Timeline and team

**Gantt (FYP-1 weeks 0–17, FYP-2 weeks 17–35):**

| Task | Weeks |
|---|---|
| Literature review | 0–2 |
| Design | 2–6 |
| Pose & spine | 5–11 |
| EMG hardware | 7–11 |
| Data collection & sync | 11–17 |
| FYP-1 demo | 13–17 |
| ML/DL | 17–25 |
| Evaluation | 24–28 |
| Testing | 27–33 |
| Documentation | 31–35 |

No slot yet for the app or real-time coaching.

**Team split (deck):**
- **All three:** literature review, design, ML/DL, documentation.
- **sEMG:** Aayan, Wajid.
- **CV + sync:** Ahmed, Aayan.
- **Testing:** Ahmed, Wajid.

The rubric wants equal contribution per phase. The team member working with Claude did the literature review.

---

## 10. Open decisions

| Decision | Status |
|---|---|
| Name | **MyoMotion** (settled) |
| Actors | **One: the User (Athlete)** (settled). No Researcher actor, role or class. Use case and deployment diagrams show only the released app |
| Live vs upload | **Both** (settled) |
| Pose pipeline | **YOLOv5 + MediaPipe** (from Ko et al., per speaker notes) |
| Exercise list | **Not final.** Many exercises covering upper body, lower body and trunk; chosen after research (criteria in §2). The squat is only the worked example |
| Camera view | Per exercise: side view where back rounding and depth matter, front view where needed (e.g. knees moving inward) |
| Spine method | Rule: SpinePose if real-time on a phone, else trigonometric; needs a side view for exercises where back rounding matters, and marker ground truth |
| sEMG channels | **3** (settled); 3 muscles per recording block, chosen per exercise (squat example: glute max, vastus lateralis, lumbar erector spinae) |
| App platform | Mobile or web + cloud (SRS); the 33 ms limit needs pose on the device |
| Asymmetry | Not measurable with this setup; drop or add a left/right block |
| Activation Efficiency Score | The UI shows it **live, per selected muscle, as 0–100** with bands "Good" (e.g. 84) and "Fair" (e.g. 64). **Formula and band cut-offs are still undefined.** |
| Two different numbers in the UI | Analysis shows **% per muscle** (Glutes 72%); Live shows an **efficiency score /100** (Glutes 84). Define both (is % = %MVC?) and explain them in the ⓘ tooltip |
| Live lean threshold | UI rule: **trunk lean > 30° → "Chest up" cue** (spoken). Make the threshold per exercise and data-driven; lean alone can't tell a hip hinge from a rounded back, so use SpinePose lumbar flexion when available |
| Muscles shown in the UI | For the squat example, Analysis lists **5** (quadriceps, glutes, lower back, hamstrings, calves) but the squat block measures 3. Show only muscles that were measured for that exercise, or record extra blocks |
| "Knees moving inward" issue | Needs a **front view**; the squat example films from the **side**. Drop it, or allow a front view for that exercise |
| Start/stop at 2–3 m | The user stands 2–3 m from the phone. Add a countdown or voice start, plus auto-stop after N reps or idle time |
| Three record entry points | "Record a set" (hero), the centre camera button and "Start live video". Define which is record-then-analyse and which is live |
| Best-posture objective (what cues coach toward) | Target-muscle share vs compensation, within safe posture limits, learned from the team's dataset |
| Model | Open; attention layer planned; LOSO. Also open: one model per exercise, or one model for all exercises with the exercise as an input |
| Accuracy target | r ≥ 0.80 LOSO (ambitious) |
| External load | Bodyweight only, load as an input, or a stated limitation |

---

## 11. FYP-1 mid-defense rubric (30 of FYP-1's 100 marks: supervisor 10.5, jury 19.5)

| CLO | Marks | Graded | Status |
|---|---|---|---|
| CLO1 Defend idea | 7 | Table/diagram with key results, gap, clear and recent references, refined problem statement | **Done** (literature deck + gap slide); fix the items in §7.3 |
| CLO2 Design & implement (agile, 20% modules) | 11 | **Analysis (0–4):** use cases, SSD, SRS, test plan, development plan, wireframes, UI. **Design (0–4):** class/domain, component, layer diagrams, DFD, data dictionary, activity, state machine, sequence. **Data (0–3):** ER/schema, data structures, algorithms | Have: SRS (needs v2), use case (needs fixes), class + sequence (MyoMotion), swimlane and activity diagrams, **final UI design / wireframes (9 screens, §13)**. Missing: SSD, test plan, development plan, remaining screens (§13.5), component diagram, rep-phase state machine, DFD, data dictionary, ER/schema (the simplified class diagram is close), algorithm pseudocode, **20% working code** |
| CLO4 Management | 6 | Updated timeline, 20% report, presentation, test cases and results, risk | Missing: updated Gantt, risk register, test cases |
| CLO5 Teamwork | 6 | Equal contribution per phase | Rebalance the table |

**Suggested 20% module:** YOLO + MediaPipe + angles + trig spine on video of one exercise (e.g. the squat) with overlay, plus 3-channel sEMG streaming with a live plot.

**Suggested tests:**
- video validation
- landmark detection rate
- angle error vs a goniometer
- SNR check
- sync offset
- FPS and latency

---

## 12. Glossary
- **sEMG**, **MVC**, **%MVC**. **Envelope:** rectified, smoothed EMG (what the AD8226 outputs).
- **X(t):** per-frame features. **Y(t):** measured %MVC. **Ŷ(t):** predicted.
- **Phases:** ECCENTRIC, ISOMETRIC_HOLD, CONCENTRIC, REST.
- **Compensation:** a non-target muscle taking over. **Activation efficiency:** target-muscle share (formula TBD). The UI shows it live as a 0–100 score for the selected muscle, with Good/Fair bands (§13).
- **UI phase names:** Down = ECCENTRIC, Hold/Bottom = ISOMETRIC_HOLD, Up = CONCENTRIC.
- **UI muscle names:** Quads/Quadriceps = vastus lateralis (sensor site), Glutes = gluteus maximus, Lower back = lumbar erector spinae.
- **LOSO:** leave-one-subject-out. **SENIAM:** sEMG placement standard. **AP:** average precision (keypoints).

---

## 13. UI design — "MyoMotion – Final UI Design" board (final wireframes)

The board header reads "FYP · Ahmed Nawaz, Aayan Sethi, Wajid Ali · Supervisor: Dr. Muhammad Amin". It has 9 iPhone-style screens (status bar 9:41, home indicator) in 3 groups. Blue connector arrows show the prototype flow.

**Scope note:** the wireframes use the **squat as the example exercise**. The screens themselves are per exercise: the Live video screen already has an **Exercise** dropdown, so "Squat analysis" becomes "<exercise> analysis". The welcome and home text ("Check your squat form", "Ready to check your squat?") will need generic wording once more exercises are added.

### 13.1 Design system (sampled from the board)
| Token | Colour | Used for |
|---|---|---|
| Primary blue | `#1F4FE0` | Primary buttons, "New analysis" card, active tab and nav item, camera button, links, "Up" phase |
| Lime accent | `#C8F43A` | Skeleton joints, "Record a set" button, "Ready" dot, "Aim here" guide line, shoes in the illustration |
| Activation magenta | `#C92A6E` | Muscle bars, efficiency score number, selected-muscle dot |
| Warning amber | `#F5A524` | Lean chip when over the limit, "Lower back" bar, live feedback banner, trunk segment when leaning |
| Amber text / background | `#864500` / `#FFF3DC` | "2 issues" chip, "Higher than expected", form issue cards |
| Success green | `#378A58` | "Good" chip, ✓ "Full body in frame", ✓ "Good depth" |
| Record red | `#EF4136` | Record/stop button, "Recording" pill and timer dot |
| Dark video background | `#151925` / `#1C2130` | Video player and live camera screens |
| Phase colours | Down `#A9BFFF`, Hold `#CDD2DE`, Up `#1F4FE0` | Phase bars |

- **Type:** geometric sans (looks like Poppins); bold headings.
- **Shapes:** rounded cards (~16 px), pill chips, full-width primary buttons.
- **Imagery:** white skeleton bones with lime joints; the trunk segment turns amber when lean is too high, with a dashed vertical reference line.
- **Tags:** "Estimated" tags and "Estimated from your video ⓘ" signal that numbers are model estimates.

### 13.2 Screen-by-screen inventory

**Group 1: User flow**

1. **Welcome** (onboarding page 1 of 3)
   - App logo (blue tile with pulse icon) + "MyoMotion"; **Skip** (top right).
   - Illustration: phone on a tripod filming a squatting person from the side, dashed field-of-view lines, skeleton overlay.
   - Title "Check your squat form with just your phone". Body: "Film one set from the side. MyoMotion tracks your posture and shows which muscles did the work."
   - 3 page dots (page 1 active; pages 2–3 not designed).
   - **Get Started** (primary; prototype links it to Login).
   - **I already have an account** (text link).

2. **Login**
   - Back arrow; "Welcome back" / "Log in to see your squat reports."
   - **Email** field (mail icon; focused; "ahmed.nawaz@gmail.com").
   - **Password** field (lock icon, masked, **eye toggle** to show/hide).
   - **Forgot password?** link; **Log in** (primary → Home).
   - Footer "New to MyoMotion? **Create account**".

3. **Home**
   - "Good evening, Ahmed" / "Ready to check your squat?"; **avatar "AN"** (top right).
   - **New analysis** hero card: "Prop your phone 2–3 m away and film one set from the side." **Record a set** (lime, camera icon); squat-skeleton art.
   - **Last session** + **See all**.
   - Session card (tap → Squat analysis):
     - skeleton thumbnail; "Squat · 3 reps"; "Today, 6:40 PM"; chip "2 issues"
     - "Muscles worked": Quads 84%, Glutes 72%, Lower back ⚠ 66% (amber)
   - **More ways to start**:
     - **Start live video** ("Real-time feedback while you squat", camera icon, chevron)
     - **Upload a video** ("MP4 or MOV from your gallery", upload icon, chevron)
   - **Bottom navigation:** Home (active) · Reports · **centre raised camera button** · History · Profile.

4. **Squat analysis: Muscles tab**
   - Header: back arrow, "Squat analysis", "Today, 6:40 PM", **share** icon.
   - **Video player:**
     - skeleton replay
     - chips "Lean 37°" (amber, top left) and "Rep 2 of 3" (top right)
     - **play** button, "0:06 / 0:14"
     - scrub bar with blue progress and **rep markers**
   - Summary: "3 reps · 4.2 s per rep · 2 issues"; "Good depth on every rep, but you leaned forward on reps 2 and 3."
   - **Tabs:** **Muscles** | Phases | Form [badge 2].
   - Muscles content: "Estimated from your video ⓘ"; bars:
     - Quadriceps 84%
     - Glutes 72%
     - Lower back ⚠ "Higher than expected" 66% (amber)
     - Hamstrings 45%
     - Calves 33%
   - Bottom nav with **Reports** active.

**Group 2: Analysis tabs** (same header, video and summary as screen 4)

5. **Phases tab**
   - "Time in each phase", legend Down / Hold / Up.
   - Stacked bars: Rep 1 = 4.1 s, Rep 2 = 4.3 s, Rep 3 = 4.3 s (average 4.2 s matches the summary).
   - "Main muscle in each phase": Going down → Quadriceps; Bottom (hold) → ⚠ Lower back; Coming up → Quadriceps.

6. **Form tab**
   - Amber issue cards, each with ⚠, a title, a fix and a **▶ play** button:
     - "Leaning forward too much — Keep your chest up. Your lower back did extra work." "Seen in reps 2 and 3 · **Show me**"
     - "Knees moving inward — Push your knees out over your toes." "Seen in rep 3 · **Show me**"
   - Positive item: ✓ "**Good depth** on all 3 reps. Nice work."

**Group 3: Live video** (board caption: "tap the red button to record, feedback appears while you squat")

7. **Live video: Ready**
   - Full-screen dark camera view; back arrow; "Live video" / "Squat · side view"; pill "● Ready" (lime).
   - Rule-of-thirds grid; overlay chips "Lean 2°", "● 00:00", "Rep 0".
   - Standing skeleton; ✓ "Full body in frame".
   - **Bottom sheet** (drag handle):
     - **Muscle** dropdown (● Glutes); **Exercise** dropdown (dumbbell icon, Squat)
     - card "**Muscle activation efficiency**" [Estimated]: "– – /100", chip "Not recording", empty sparkline, "Starts when you tap record"
   - Controls: **recordings/gallery** button (left), big **red record** button "Tap to record" (centre), **flip camera** (right).
   - No bottom nav.

8. **Live video: Recording**
   - Pill "● Recording" (red); "Lean 27°"; timer "● 00:01"; "Rep 1".
   - Squatting skeleton.
   - Score **84 /100** + chip "**Good**"; sparkline; "Glutes · live estimate from video".
   - **Stop** button (red square in a ring) "Tap to stop".

9. **Live video: Feedback**
   - "Lean 36°" chip turns amber; "00:05"; "Rep 2".
   - Trunk drawn amber, with a **dashed lime target line labelled "Aim here"**.
   - **Live feedback banner** (amber) **replaces the Muscle/Exercise row**: ⚠ "Live feedback · Rep 2", "**Chest up**", "Lift your back to the dashed line", **speaker icon** (spoken aloud).
   - Score **64 /100** + chip "**Fair**"; sparkline drops; "Leaning forward takes work away from your glutes".

**Board caption rule:** opened from Home > More ways to start > Start live video. The red button starts recording. The skeleton follows the squat and the efficiency score updates live. **If the back leans past 30°**, live feedback replaces the Muscle/Exercise row ("Chest up", also spoken), a dashed line shows where the back should be, and the score drops.

### 13.3 Navigation map
**Prototype links shown:** Get Started → Login; Log in → Home; Last-session card → Squat analysis.

**Implied links:**
- **Welcome:**
  - Skip → Home (guest?)
  - I already have an account → Login
- **Login:**
  - Create account → sign-up
  - Forgot password? → reset
  - back → Welcome
- **Home:**
  - Record a set → camera
  - See all → History/Reports
  - Start live video → Live video
  - Upload a video → gallery picker → processing → Squat analysis
  - AN → Profile
  - nav tabs → Home / Reports / History / Profile; centre camera button → recording
- **Squat analysis:**
  - tabs switch content
  - ▶ / Show me → jump the replay to that rep
  - share → export (SRS FR-UI-003: PDF; annotated video)
- **Live video:**
  - record → recording
  - stop → (should open Squat analysis, not drawn)
  - gallery → recordings
  - flip camera
  - back

### 13.4 What the backend must supply (data contract implied by the UI)
- **Session:**
  - exercise, timestamp, video URL and duration (0:14)
  - rep count (3); average rep time (4.2 s)
  - issue count (2); one-line summary text
- **Per rep:** phase durations (down/hold/up) and lean angle; which reps each issue occurred in.
- **Per muscle:** estimated % (bar), "higher than expected" flag (compensation vs an expected range), and the main muscle per phase.
- **Form issues:**
  - type, title, fix text, rep list
  - a time offset so "Show me" can jump the replay
  - positive findings ("Good depth")
- **Live, per frame:**
  - skeleton points (+ spine), lean angle, rep counter
  - full-body-in-frame flag
  - efficiency score for the selected muscle (0–100) + band (Good/Fair)
  - sparkline history
  - active cue (title, fix, target line geometry) + text-to-speech
- **User:** first name (greeting), initials (avatar).
- **Class diagram fit:** the simplified class diagram has no **FormIssue/Feedback** entity yet. Add one (type, message, fix, reps, severity), plus **per-rep phase durations** and lean.

### 13.5 Gaps and issues in the wireframes (to fix or design next)
1. **Screens not drawn:**
   - onboarding pages 2–3
   - Create account; Forgot password
   - Upload flow + processing/loading state
   - post-recording result
   - Reports, History, Profile/settings (including an audio-cue toggle)
   - share/export sheet (PDF / annotated video)
   - camera-permission screen
   - "**not in frame**" warning state (sequence diagram: warn after 1 s)
2. **"Knees moving inward"** can't be seen from a **side view** (squat example). Drop it or allow a front view for that exercise.
   - Squat-specific wording ("Check your squat form", "Ready to check your squat?", "Squat analysis") should become generic as exercises are added.
3. **The user is 2–3 m from the phone**, so "Tap to record/stop" is impractical. Add a 3-2-1 countdown, voice start, and auto-stop after N reps or idle time.
   - The SRS requires feedback **legible from 2 m**: make the live cue text and score much larger and keep the spoken cue.
4. **Two numbers can confuse users:**
   - Analysis % per muscle (Glutes 72%) vs live efficiency score (Glutes 84/100).
   - Define both; say whether % means **%MVC**; use the ⓘ tooltip; define the Good/Fair/(Poor) bands.
5. **Five muscles shown vs 3 sensors** (glute, vastus lateralis, erector spinae). Estimating hamstrings and calves needs Block B data; otherwise show only measured muscles.
6. **Naming:** "Quads" on Home vs "Quadriceps" in Analysis; "Lower back" in the UI = erector spinae in the docs. Keep one mapping.
7. **Lean > 30° rule:** lean alone can't separate a hip hinge from a rounded back (Straub & Powers). Use SpinePose lumbar flexion when available.
8. **Three record entry points** ("Record a set", the centre camera button, "Start live video"): define which records-then-analyses and which is live.
9. **Flow arrow:** Get Started → Login. For new users it should go to Create account; "I already have an account" → Login.
10. **Phase names:** the UI uses Down/Hold/Up; the model uses ECCENTRIC/ISOMETRIC_HOLD/CONCENTRIC (+REST). Keep that mapping.

### 13.6 How the UI maps to the docs
- **SRS:**
  - FR-UI-001 (overlay): live and replay skeleton
  - FR-UI-002 (cues): Form tab + live banner
  - FR-UI-003 (export): share icon, sheet not drawn
  - FR-ML-002 (compensation): "Lower back – Higher than expected"
  - FR-ML-003 (efficiency): live score
  - Phase segmentation: Phases tab and rep markers
- **Sequence diagram:** matches the live flow (pose → spine → features → phase → activation → form → overlay/cue; warning when not in frame).
- **Rubric (CLO2 System Analysis):** "Wireframes, UI screens" is now covered by this board.
