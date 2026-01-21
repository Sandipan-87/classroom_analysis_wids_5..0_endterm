# Classroom Attention Analytics: A Human-Aware Computer Vision


This repository represents the final evolution of that journey: a **"Conservative & Human-Aware"** system. It moves away from strict surveillance logic and adopts a teacher's intuition: **assume focus until proven otherwise.**

---

## The Architecture

### 1. YOLOv8 & MediaPipe
We use a hybrid pipeline to balance speed and accuracy:
* **YOLOv8s** acts as the "Scene Detector," identifying every student in the frame.
* **Hungarian Algorithm** tracks these detections across frames, assigning unique IDs (e.g., `S01`, `S02`) that persist even if detection flickers.
* **MediaPipe Face Mesh** is the precision instrument. It extracts 468 facial landmarks to compute the **3D Head Pose (Yaw & Pitch)**.

### 2. `AttentionAnalyzer`
This is where the logic lives. Instead of simple `if/else` statements, I built a state machine with "inertia":
* **Green States (`FOCUSED`, `TAKING_NOTES`)** are "sticky." Once a student enters this state, the system resists kicking them out for minor movements.
* **Yellow State (`LOOKING_AWAY`)** is treated as transient. A quick glance at a neighbor isn't distraction; it's normal behavior.
* **Red State (`DISTRACTED`)** is the sink. It only triggers after sustained disengagement.

### 3. Dynamic Board Calibration
One of the hardest geometric problems was the "Angle of Incidence." A student sitting in the far left corner *must* turn their head right to see the board. A naive algorithm sees `Yaw: +40°` and thinks "Looking Away."
* **Solution:** The `BoardCalibrator` spends the first 5 seconds "learning" the room. It calculates the average gaze vector of the class and sets that as the "Zero Point." All attention is calculated relative to this dynamic "Board North."

---

##  The Struggles

Building this wasn't smooth. Here are the dragons we had to slay:

### 1. The "Dependency Hell"
Getting `ultralytics` (YOLO), `mediapipe`, `opencv`, and `numpy` to coexist is a nightmare, especially in Google Colab. Runtime disconnects, version conflicts, and missing DLLs were constant battles.
**The Fix:** Actually I started coding this on colab but then I switched it on VS code on a venv of python version 3.10.9.

### 2. The "Statue" Problem 
In early versions, we required "Stability" (low variance) to mark someone as focused. The result? **Diligent students typing notes were marked "Distracted"** because their heads were moving. We called this "Green Starvation"—the system refused to award credit for active work.
**The Fix:** I implemented the **"Strong Focus Override."** If a student's nose points dead-center at the board, we ignore stability checks entirely. Geometry trumps variance.

### 3. The "Flicker" Effect
Neural networks are noisy. A face might detect as `Yaw: 20°` in one frame and `28°` in the next. This caused students to flicker rapidly between "Focused" (Green) and "Looking Away" (Yellow).
**The Fix:** We introduced **Analog Timer Decay**. Instead of resetting focus timers to 0.0 instantly upon a mistake, they decay slowly. This gives the system "memory" and smooths out the noise.

---

## Bias Configuration

**Current Tuning: High Focus Bias**
To counter the "Statue Problem," this version is intentionally biased toward **Green (Focus)**. We assume students are paying attention unless they prove otherwise.

You can tweak this "Teacher's Personality" in the `CONFIG` section of `main.py`:

```python
CONFIG = {
    # === SENSITIVITY CONTROLS ===
    
    # The "Field of View"
    # Current: 45° (Very Generous). Matches a wide whiteboard.
    # To make stricter: Lower to 25°
    "focused_yaw_max": 45,
    
    # The "Fidget Factor"
    # Current: 25.0 (High Tolerance). Allows typing/nodding.
    # To make stricter: Lower to 10.0
    "stable_variance_max": 25.0,
    
    # The "Benefit of Doubt"
    # Current: 0.1s (Instant). Credit is given immediately.
    # To make stricter: Increase to 0.5s
    "focused_min_duration": 0.1,
}
```

## Limitations & Flaws
Here is where the system still struggles:

* **The "Sleeping Student" Loophole:** The system tracks **Head Pose**, not **Eye Gaze**. A student can face the board, close their eyes, and sleep. The system will mark them as "Focused."
* **ID Swapping:** If two students walk past each other or sit very close (overlapping bounding boxes), the tracker might swap their IDs (e.g., `S05` becomes `S08`).
* **2D-to-3D Estimation:** We estimate 3D angles from a flat 2D image. Extreme camera angles or hands covering the face can cause the "Pitch" calculation to jump wildly.
* **No Audio Context:** A loud, disruptive class might be visually "Focused" (facing forward) but acoustically "Distracted." The system is deaf.

## Scope for Future Improvement
* **Gaze Tracking Integration:** Adding an eye-tracking submodule would solve the "Sleeping Student" loophole by verifying if eyes are open and looking at the board.
* **Re-Identification (ReID):** Implementing a visual feature extractor (like ResNet) to memorize student clothing/appearance. This would allow `S01` to leave the room, come back, and still be `S01`.
* **Seat Mapping:** Instead of random IDs, the system could automatically generate a "Seating Chart" (e.g., `Row 1, Seat A`) for better reporting.
* **Real-Time Dashboard:** Replacing the Excel export with a live web dashboard (Streamlit/Flask) for real-time teacher feedback.

## Outputs & Reports
The script produces three key artifacts:

1.  **`attention_TIMESTAMP.mp4`**: A visual overlay showing the "Thought Process" of the AI (bounding boxes, state labels, debug stats).
2.  **`attention_report_TIMESTAMP.xlsx`**: A raw data dump containing second-by-second timelines and aggregate student scores.
3.  **Visual Plots:** (Generated via `plot.py`) A visual dashboard of the class timeline, showing the rise and fall of attention over the lecture.
