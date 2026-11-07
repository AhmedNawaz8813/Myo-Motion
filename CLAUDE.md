# CLAUDE.md — MyoMotion (BSCS Final Year Project)

You are helping a three-person BSCS team at FAST-NUCES Peshawar with their FYP.
- **Full knowledge base:** `Context.md` (pipeline, hardware, SRS, UML, literature deck with speaker notes, rubric, open decisions, UI wireframes). Read it before substantial work.
- If you only have this file, the summary below is enough to start.

@Context.md

## The project in one paragraph
**MyoMotion** is an app that coaches a user through **many exercises that together work muscles across the whole body** (upper body, lower body and trunk), from camera video. It checks posture, estimates how much each target muscle is working (%MVC) in each rep phase, and gives corrective cues.
- **Exercise list: not final.** The team will choose the exact exercises after more research (Context.md §2). The **squat is only the worked example** used so far in the wireframes, diagrams and literature deck. Never describe MyoMotion as squat-only.
- **For the user it is camera-only:** prop a phone 2–3 m away and film the set from the view the app asks for that exercise. No sensors on the user.
- **sEMG is the team's research tool:** the team records synchronised video + sEMG from participants to build the training dataset. The model learns to estimate activation from video alone.
- **Input:** live coaching (≤ 33 ms per frame) and uploaded video.
- **Tech:**
  - YOLOv5 + MediaPipe Pose + spine module (SpinePose or trigonometric)
  - Arduino Uno + 3 × AD8226 EMG modules (dataset collection)
  - ML/DL regression X(t) → %MVC, evaluated leave-one-subject-out
- **Team:** Ahmed Nawaz, Aayan Sethi, Wajid Ali. **Supervisor:** Dr. Muhammad Amin.
- **Stage:** FYP-1; preparing for the mid defense; literature review done; final UI wireframes done (9 screens; Context.md §13).

## Names and terms (use exactly)
- **Project name:** **MyoMotion**. Older files say "AIFIT"/"AI-FIT" (interim name); don't reuse that name.
  - *AIFit* (Fieraru et al., CVPR 2021) is a separate paper the team cites.
- **sEMG**, **MVC** (never "MCV"), **%MVC**.
- **Signals:** X(t) = per-frame features; Y(t) = measured %MVC; Ŷ(t) = predicted.
- **Rep phases:** ECCENTRIC, ISOMETRIC_HOLD, CONCENTRIC, REST (UI: Down, Hold, Up).
- **Muscle names (squat example):** Glutes = gluteus maximus; Quadriceps = vastus lateralis (sensor site); Lower back = lumbar erector spinae. Other exercises add their own muscles; keep one display name per muscle.
- **Actors:** only one, the **User (Athlete)**. There is **no Researcher** actor, role or class anywhere (the team collects the dataset, but that is not an app feature). Never add a Researcher to a diagram.

## Facts that must stay true
- **MediaPipe has 33 landmarks and no spine points**; the only trunk points are the shoulders (11, 12) and hips (23, 24).
  - Spine rule (team's speaker notes): **SpinePose** (9 spine keypoints, 2D) if it gives real-time feedback on a phone; otherwise **trigonometric** shoulder/hip proxies, which give trunk lean only.
  - SpinePose is 2D, so film from the side when back rounding matters. The camera view is set per exercise (side or front).
  - Hip/shoulder-only trunk keypoints gave 12–39° errors in OpenCap.
- **3 × AD8226 modules output a rectified, smoothed envelope**, not raw EMG.
  - Normalise to MVC only.
  - Fatigue from amplitude only (no frequency analysis).
  - ±9V from two 9V batteries (shareable).
  - Gain set by a potentiometer. Sample 500 Hz–1 kHz.
- **3 sensors = 3 muscles per recording, one side.**
  - Muscles are chosen per exercise (target muscle, posture-sensitive muscle, likely compensator; SENIAM sites). Squat example: gluteus maximus, vastus lateralis, lumbar erector spinae.
  - More exercises across the body means more recording blocks per participant (electrodes moved between blocks, MVC for each muscle).
  - **No bilateral asymmetry** without a dedicated left/right block.
- **Recording-session rules:**
  - Redo MVC every session.
  - Python serial logger (not the Serial Plotter).
  - LED + timestamp sync.
  - Laptop on battery or a USB isolator while electrodes are on.
  - Record several posture variations of each exercise per participant (squat example: stance width × foot angle × depth) at the same load and tempo, in random order, with rest and several reps each.
- **Each exercise needs its own recorded data:** MuscleMap's accuracy fell from 97.8% to 64.2% mAP on activities it had not seen, so the exercise list drives dataset size. Add exercises in stages; retraining one layer (Lehmler) can make that cheaper.
- **More activation is not automatically better.** The objective is target-muscle share vs compensating muscles, within safe posture limits.
- **Live mode** needs causal models (attention looks at past frames only) and on-device pose. Cloud-per-frame can't meet ≤ 33 ms.
- **Prior work:** *Muscles in Action* and *OpenCap* already estimate muscles from video.
  - Never claim nobody links video to muscle activity.
  - MyoMotion's gap: spine detail + rep phases + posture cues tied to muscle activation + live coaching from one phone camera.
- **Not a medical device.**

## App UI (final wireframes; full inventory in Context.md §13)
- **Screens (9):**
  - User flow: Welcome (onboarding 1/3), Login, Home
  - Analysis tabs: "Squat analysis" (the squat is the example; the screen is per exercise) with tabs **Muscles | Phases | Form**
  - Live video: **Ready → Recording → Feedback**
- **Flow:**
  - Welcome → Login → Home
  - Home "Last session" card → Squat analysis
  - Home → "More ways to start" → **Start live video** / **Upload a video** (MP4/MOV)
  - Bottom nav: Home · Reports · centre camera button · History · Profile
- **Design tokens:**

  | Use | Colour |
  |---|---|
  | Primary blue | `#1F4FE0` |
  | Lime accent (joints, record CTA, target line) | `#C8F43A` |
  | Activation magenta (bars, score) | `#C92A6E` |
  | Warning amber | `#F5A524` (text `#864500`, card `#FFF3DC`) |
  | Success green | `#378A58` |
  | Record red | `#EF4136` |
  | Dark video background | `#151925` |
  | Phases Down / Hold / Up | `#A9BFFF` / `#CDD2DE` / `#1F4FE0` |

  Poppins-style sans; rounded cards; pill chips.
- **UI rules to keep:**
  - phone **2–3 m** away; side view in the squat example (camera view set per exercise); the Live video screen already has an **Exercise** dropdown
  - skeleton overlay with lime joints; trunk turns amber when leaning
  - **lean > 30° → live "Chest up" cue** (banner replaces the Muscle/Exercise row, spoken aloud, dashed "Aim here" line, score drops)
  - every model number is tagged **"Estimated"**
  - Form issues are cards with a fix, the reps they occurred in, and **Show me** (jumps the replay)
- **Known UI gaps:**
  - no sign-up / forgot-password / upload-processing / Reports / History / Profile / export sheet / not-in-frame state
  - "Knees moving inward" can't be seen from a side view (needs a front view for that exercise)
  - 5 muscles shown vs 3 sensors
  - % (analysis) vs /100 (live) undefined
  - tap-to-record at 2–3 m needs a countdown/voice start and auto-stop
- **When designing new screens:** reuse the tokens and components above, and keep naming consistent (Quadriceps, Glutes, Lower back).

## How to help
- **Open decisions:** check "Open decisions" in Context.md before assuming. Ask, or state the assumption. Update Context.md and its date when the team decides something.
- **Rubric:** prefer outputs that score on the FYP-1 mid rubric (CLO1 7, CLO2 11, CLO4 6, CLO5 6; Context.md §11).
- **Diagrams:**
  - **PlantUML**: render PNG + SVG + `.puml` and check the render.
  - Current style: plain black/grey/white via `diagrams/mono_style.iuml`; the team edits diagrams in **Software Ideas Modeler** (provide `.simp` where possible, see Context.md §6.1).
  - Class diagram: domain model, attribute : type, `{field}` prefix, no methods, no FKs (keys live in the ERD), multiplicities on every association, composition (◆) for whole-part relations.
  - ERD: crow's foot, PK/FK marked, app tables + training dataset tables (no link to users, no role column).
  - Sequence diagrams: alt/opt/loop fragments with guards; keep names consistent with the class diagram.
  - Use case and deployment diagrams: **released app features only**, nothing from building the app (data collection, training).
- **UML rules:**
  - use case: the one primary actor (User) on the left; secondary actors (external systems the app calls: Device Camera, Text to Speech, Phone Gallery, Email Service) on the right; the MyoMotion server is inside the system, not an actor
  - solid actor–use case lines
  - correct «include»/«extend»
  - guards on every decision
  - retry paths loop back
- **Citations:** IEEE. The deck uses [1]–[11] (Context.md §7.4). Never invent citations; open the paper's page before citing it and take numbers from it.
- **SRS:** fix the known defects (Context.md §5.3) when editing; use "shall"; every acceptance-table ID must exist in Section 3.
- **Code:** Python (YOLOv5, MediaPipe, scikit-learn/PyTorch) and Arduino C++. The app is mobile or web.
- **Style:** direct, practical for three undergrads; flag problems briefly. The team likes reusable files (`.puml`, `.pptx`, `.xlsx`, `.md`).

## Key files
| File | What it is |
|---|---|
| `MyoMotion_Literature_Review.pptx` | Literature review: 11 papers in 5 table slides (with a Relevance to MyoMotion column) and speaker notes, comparison matrix, Research Gap & Contribution, Problem Statement, MyoMotion's Solution (YOLO + MediaPipe pipeline), 2 reference slides. Team style: plain black/grey/white, bold-label text, no cards or coloured callouts |
| UI board PNG ("MyoMotion – Final UI Design") | Final wireframes: 9 phone screens (Context.md §13) |
| `diagrams/` (+ `MyoMotion_Diagrams.zip`) | **Current diagram set v3 (2026-10-12), exercise-generic, one actor (User), no Researcher:** use case (released app; User on the left, secondary system actors on the right); activities (upload, live); sequence (live coaching); class (app only); ERD (app tables + training dataset tables, no link to users); architecture (the only diagram with the sEMG setup); deployment (released app). Each as `.puml/.png/.svg`, all 8 in `MyoMotion_Diagrams.pdf`; SIM files in `diagrams/sim/` (no deployment); `README.md` lists what to fix in SIM |
| `MyoMotion_Class_Diagram.*`, `MyoMotion_Sequence_Diagram.*` (root) | Earlier versions, superseded by `diagrams/` |
| `AIFIT_Class_Diagram.*`, `AIFIT_Sequence_Diagram.*`, `AIFIT_Swimlane_Diagram.*` | Older detailed design set; outdated, don't reuse |
| `AIFIT_Literature_Tracker.xlsx`, Claude Docs "AIFIT Literature Review" | Extended 26-paper literature material |
| `FYP_SRS.pdf` | 6-page SRS (needs v2) |
| `FYP1_Proposal_Presentation.pptx` | 14-slide proposal defense deck |
