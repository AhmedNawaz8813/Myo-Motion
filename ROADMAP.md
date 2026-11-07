# MyoMotion: Project Timeline & Development Roadmap

Welcome to the MyoMotion project repository. This document outlines our development roadmap, from initial pipeline prototyping to the final deployment of our biomechanics and muscle activation coaching application. It is designed to give contributors and users a clear understanding of our trajectory.

## Phase 1: Pipeline Prototyping & Individual Testing
Before integrating the system, we will develop and rigorously test our primary data pipelines in isolation.

*   **Computer Vision Pipeline:** We will build and test the camera-based tracking system. This involves deploying YOLOv5 for person detection, MediaPipe for 33-point pose estimation, and our dedicated spine module (using SpinePose or trigonometric proxies) to accurately track joint angles, trunk lean, and posture variations.
*   **Hardware & sEMG Pipeline:** In parallel, we will test our hardware stack (Arduino Uno with 3 × AD8226 sEMG modules). This phase focuses on reliable data acquisition, capturing Maximum Voluntary Contraction (MVC), and establishing a robust pipeline for cleaning data. This includes rectifying and smoothing the envelope signal and normalizing it to %MVC without losing critical amplitude fatigue indicators.

## Phase 2: System Synchronization
Once the individual pipelines are stable, we will develop a unified pipeline that synchronizes both data collection methods. Using an LED/beep timestamp sync mechanism, we will ensure that the high-framerate visual data ($X(t)$) aligns perfectly with the sEMG signal data ($Y(t)$). This synchronization is critical to establishing a reliable ground truth for our machine learning models.

## Phase 3: Field Data Collection
With the synchronized DAQ (Data Acquisition) pipeline ready, we will move into physical gyms to collect our primary dataset. 

*   **Target Exercises:** We will capture the three primary powerlifting movements: the Squat, Bench Press, and Deadlift. 
*   **Protocol:** For each exercise, we will record synchronized visual and sEMG data across multiple participants. The protocol will include various posture variations (e.g., stance width, foot angle, depth) while keeping load and tempo consistent. This ensures our models learn exactly how slight postural shifts affect target-muscle share and compensation.

## Phase 4: Data Processing, Feature Engineering & Model Training
Following data collection, we will prepare the data for our ML/DL models.

*   **Feature Engineering:** We will clean the data further, mapping visual features (joint angles, spine curvature) to their respective repetition phases (Eccentric, Isometric Hold, Concentric).
*   **Model Training:** We will train a variety of predictive models to perform the $X(t) \rightarrow Y(t)$ regression. This will include traditional Machine Learning algorithms, deep learning approaches like CNNs and LSTMs, and our own custom neural network architecture (incorporating attention layers to weight critical frames).
*   **Evaluation:** Models will be evaluated using strict Leave-One-Subject-Out (LOSO) cross-validation to ensure the system generalizes well to unseen users, targeting a high correlation coefficient ($r \ge 0.80$).

## Phase 5: Dual-Variant Product Development
Based on our trained models, we will produce two distinct variants of the MyoMotion product to cater to different user needs:

1.  **The Global Model (Vision-Only):** The primary, highly accessible variant of the app. It will require only a smartphone camera (no sensors attached to the user). It will take in visual features and output estimated muscle activation and posture cues in real-time or via offline batch uploads.
2.  **The Hybrid Model (Hardware-Calibrated):** A premium or research-focused variant that includes a brief, initial sEMG hardware calibration session. By taking live baseline readings to enhance and anchor the sEMG signal output accuracy, this model will provide hyper-personalized muscle activation estimates based on subsequent visual-only inputs.

## Phase 6: Application UI/UX & Live Coaching Integration
The final technical phase involves tying the models into the MyoMotion end-user application.
*   **Interface Implementation:** Building out the planned UI screens (Live Video, Analysis, Phase Breakdowns).
*   **Real-Time Optimization:** Ensuring the on-device inference runs at $\le 33$ ms per frame to provide instantaneous, text-to-speech coaching cues (e.g., "Chest up") the moment a user breaches safe posture limits.
*   **Reporting:** Generating comprehensive post-set reports detailing activation efficiency, form issues, and rep-by-rep breakdowns.

## Phase 7: Documentation & Final Defense
The final stretch will be dedicated to polishing the repository, finalizing our Software Requirements Specification (SRS), ensuring all UML and ERD diagrams accurately reflect the deployed architecture, and preparing the complete thesis and defense presentation for our Final Year Project (FYP-2) conclusion.