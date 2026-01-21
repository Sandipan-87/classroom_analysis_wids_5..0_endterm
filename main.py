"""
Classroom Attention Analytics 
=============================

STATES (4 States):
  🟢 FOCUSED: Normal posture facing board
  🟢 TAKING_NOTES: Looking down at desk (green - productive)
  🟡 LOOKING_AWAY: Sideways from board (transient only)
  🔴 DISTRACTED: LOOKING_AWAY that persists (rare)

CORE SIGNALS:
  - Head yaw (BOARD-relative, not camera)
  - Head pitch
  - Pose stability
  - Temporal persistence
  - Nodding (positive boost)
"""

import cv2
import numpy as np
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

def check_dependencies():
    missing = []
    for pkg in ["ultralytics", "mediapipe", "scipy", "pandas", "openpyxl"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"   Missing: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from ultralytics import YOLO
import mediapipe as mp
from scipy.optimize import linear_sum_assignment
import pandas as pd

# ============================================================================
# CONFIGURATION - CONSERVATIVE PHILOSOPHY
# ============================================================================

CONFIG = {
    # Input
    "video_path": "classroom_clip.mp4",
    "max_seconds": 295,
    
    # Detection
    "yolo_model": "yolov8s.pt",
    "detection_confidence": 0.25,
    "min_face_size": 25,
    
    # === BOARD CALIBRATION ===
    "calibration_seconds": 5.0,
    
    # === ANGLE ZONES  ===
    # Yaw zones 
    "focused_yaw_max": 45,           
    "looking_away_yaw_min": 45,      
    "looking_away_yaw_max": 65,      
    # Beyond 55° = extreme (fast track to DISTRACTED)
    
    # Pitch zones
    "upright_pitch_min": -15,       
    "upright_pitch_max": 35,         
    "notes_pitch_min": 35,          
    "notes_pitch_max": 50,         
    # Beyond +50° = drooping (potential sleep)
    
    # === STABILITY THRESHOLDS ===
    "stable_variance_max": 25.0,
    
    # === DURATION THRESHOLDS ===
    "focused_min_duration": 0.1,
    "notes_min_duration": 0.8,
    "looking_away_max_duration": 2.0,  # After this → DISTRACTED
    "extreme_yaw_fast_track": 1.5,     # Extreme yaw → faster DISTRACTED
    "recovery_grace_period": 1.5,      
    
    # === CROWD PRIOR ===
    "crowd_prior_threshold": 0.6,
    "crowd_yaw_bonus": 5.0,
    "crowd_duration_bonus": 0.3,
    
    # === NODDING DETECTION ===
    "nod_amplitude_min": 6,
    "nod_freq_min": 0.4,
    "nod_freq_max": 2.5,
    
    # === FACE VISIBILITY ===
    "face_missing_hold_seconds": 2.0,
    
    # === STATE WEIGHTS ===
    "weight_focused": 1.0,
    "weight_taking_notes": 0.95,  # Slightly less than focused but still green
    "weight_looking_away": 0.4,
    "weight_distracted": 0.1,
    
    # Visual
    "color_green": (0, 200, 0),
    "color_yellow": (0, 200, 200),
    "color_red": (0, 0, 220),
    "color_unknown": (128, 128, 128),
    
    # Output
    "output_video": True,
    "output_excel": True,
}

# Global FPS
FPS = 30

# ============================================================================
# STATES - 4 STATES
# ============================================================================

class AttentionState(Enum):
    FOCUSED = "Focused"              #  Normal posture, facing board
    TAKING_NOTES = "Taking Notes"    #  Looking down (productive)
    LOOKING_AWAY = "Looking Away"    #  Sideways (transient only)
    DISTRACTED = "Distracted"        #  Prolonged looking away (RARE)
    NOT_VISIBLE = "Not Visible"      #  Face not detected

STATE_INFO = {
    AttentionState.FOCUSED: {
        "weight": CONFIG["weight_focused"],
        "color": CONFIG["color_green"],
        "category": "positive",
        "display": "Focused"
    },
    AttentionState.TAKING_NOTES: {
        "weight": CONFIG["weight_taking_notes"],
        "color": CONFIG["color_green"],  
        "category": "positive",
        "display": "Taking Notes"
    },
    AttentionState.LOOKING_AWAY: {
        "weight": CONFIG["weight_looking_away"],
        "color": CONFIG["color_yellow"],
        "category": "neutral",
        "display": "Looking Away"
    },
    AttentionState.DISTRACTED: {
        "weight": CONFIG["weight_distracted"],
        "color": CONFIG["color_red"],
        "category": "negative",
        "display": "Distracted"
    },
    AttentionState.NOT_VISIBLE: {
        "weight": None,
        "color": CONFIG["color_unknown"],
        "category": "unknown",
        "display": "Not Visible"
    },
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FaceData:
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    raw_yaw: float = 0.0
    board_yaw: float = 0.0  
    pitch: float = 0.0

@dataclass
class StudentRecord:
    student_id: str
    track_id: int
    
    face_data: FaceData = field(default_factory=FaceData)
    
    current_state: AttentionState = AttentionState.NOT_VISIBLE
    previous_state: AttentionState = AttentionState.NOT_VISIBLE
    
    # Timing for state transitions
    looking_away_start_time: float = 0.0
    last_focused_time: float = 0.0
    last_notes_time: float = 0.0
    
    # Pose history
    pose_history: deque = field(default_factory=lambda: deque(maxlen=90))
    
    # Stability
    yaw_variance: float = 0.0
    pitch_variance: float = 0.0
    pose_variance: float = 0.0
    is_stable: bool = True
    
    # Nodding
    is_nodding: bool = False
    
    # Visibility
    last_visible_time: float = 0.0
    frames_not_visible: int = 0
    last_visible_state: AttentionState = AttentionState.FOCUSED
    
    # Statistics
    state_durations: Dict = field(default_factory=lambda: defaultdict(float))
    total_visible_time: float = 0.0
    focus_index: float = 0.0
    
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_frames: int = 0


# ============================================================================
# BOARD CALIBRATOR
# ============================================================================

class BoardCalibrator:
    """
    Calibrates board direction from first 5 seconds.
    All yaw measurements become BOARD-RELATIVE after calibration.
    """
    
    def __init__(self, calibration_seconds: float = 5.0):
        self.calibration_seconds = calibration_seconds
        self.calibration_samples: List[float] = []
        self.is_calibrated = False
        self.board_yaw_offset = 0.0
        self.calibration_start_time = None
    
    def add_sample(self, yaw: float, variance: float, timestamp: float):
        if self.calibration_start_time is None:
            self.calibration_start_time = timestamp
        
        if timestamp - self.calibration_start_time <= self.calibration_seconds:
           
            if variance <= CONFIG["stable_variance_max"]:
                self.calibration_samples.append(yaw)
    
    def calibrate(self) -> float:
        if not self.calibration_samples:
            print("    No stable samples, using offset = 0°")
            self.board_yaw_offset = 0.0
        else:
            self.board_yaw_offset = float(np.median(self.calibration_samples))
            print(f"    Board direction: {self.board_yaw_offset:+.1f}° from camera")
            print(f"      ({len(self.calibration_samples)} stable samples)")
        
        self.is_calibrated = True
        return self.board_yaw_offset
    
    def get_board_relative_yaw(self, raw_yaw: float) -> float:
        """Convert camera-relative yaw to BOARD-relative yaw."""
        if not self.is_calibrated:
            return raw_yaw
        return raw_yaw - self.board_yaw_offset
    
    def should_calibrate(self, timestamp: float) -> bool:
        if self.calibration_start_time is None:
            return False
        return (timestamp - self.calibration_start_time >= self.calibration_seconds
                and not self.is_calibrated)


# ============================================================================
# CROWD PRIOR
# ============================================================================

class CrowdPrior:
    """Scene-level attention prior for benefit of doubt."""
    
    def __init__(self):
        self.focused_ratio = 0.5
        self.last_update_time = 0.0
    
    def update(self, students: Dict, timestamp: float):
        if timestamp - self.last_update_time < 1.0:
            return
        
        self.last_update_time = timestamp
        
        visible = [s for s in students.values()
                   if s.current_state != AttentionState.NOT_VISIBLE]
        
        if len(visible) < 2:
            self.focused_ratio = 0.5
            return
        
        positive = sum(1 for s in visible
                       if s.current_state in [AttentionState.FOCUSED, AttentionState.TAKING_NOTES])
        self.focused_ratio = positive / len(visible)
    
    def get_yaw_bonus(self) -> float:
        if self.focused_ratio > CONFIG["crowd_prior_threshold"]:
            return CONFIG["crowd_yaw_bonus"]
        return 0.0
    
    def get_duration_bonus(self) -> float:
        if self.focused_ratio > CONFIG["crowd_prior_threshold"]:
            return CONFIG["crowd_duration_bonus"]
        return 0.0


# ============================================================================
# FACE TRACKER
# ============================================================================

class FaceTracker:
    """Face detection attached to person tracks."""
    
    def __init__(self):
        print("⏳ Loading YOLO...")
        self.yolo = YOLO(CONFIG["yolo_model"])
        print("✅ YOLO ready")
        
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=25,
            refine_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        
        self.person_tracks = {}
        self.next_person_id = 1
        self.max_lost_frames = int(CONFIG["face_missing_hold_seconds"] * FPS) + 15
        
        self.id_map = {}
        self.next_student = 1
        
        # Landmarks
        self.NOSE = 1
        self.CHIN = 152
        self.L_EYE = 33
        self.R_EYE = 263
        self.L_MOUTH = 61
        self.R_MOUTH = 291
        
        self.model_pts = np.array([
            (0, 0, 0), (0, -63.6, -12.5),
            (-43.3, 32.7, -26), (43.3, 32.7, -26),
            (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
        ], dtype=np.float64)
    
    def get_student_id(self, tid: int) -> str:
        if tid not in self.id_map:
            self.id_map[tid] = f"S{self.next_student:02d}"
            self.next_student += 1
        return self.id_map[tid]
    
    def _iou(self, b1, b2) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0
    
    def _get_head_roi(self, person_bbox, frame_shape):
        h, w = frame_shape[:2]
        px1, py1, px2, py2 = person_bbox
        head_height = int(0.35 * (py2 - py1))
        return (
            max(0, px1 - 10),
            max(0, py1 - 5),
            min(w, px2 + 10),
            min(h, py1 + head_height)
        )
    
    def _get_face_box(self, landmarks, frame_shape, roi_box):
        h, w = frame_shape[:2]
        rx1, ry1, rx2, ry2 = roi_box
        roi_w, roi_h = rx2 - rx1, ry2 - ry1
        
        indices = [10, 152, 234, 454, 33, 263, 1, 61, 291]
        pts = []
        for i in indices:
            if i < len(landmarks):
                x = int(landmarks[i].x * roi_w + rx1)
                y = int(landmarks[i].y * roi_h + ry1)
                pts.append((x, y))
        
        if not pts:
            return roi_box
        
        xs, ys = zip(*pts)
        pad_x = int((max(xs) - min(xs)) * 0.25)
        pad_y = int((max(ys) - min(ys)) * 0.2)
        
        return (
            max(0, min(xs) - pad_x),
            max(0, min(ys) - pad_y),
            min(w, max(xs) + pad_x),
            min(h, max(ys) + pad_y)
        )
    
    def _get_head_pose(self, landmarks, roi_shape) -> Tuple[float, float]:
        h, w = roi_shape[:2]
        try:
            pts = np.array([
                (landmarks[self.NOSE].x * w, landmarks[self.NOSE].y * h),
                (landmarks[self.CHIN].x * w, landmarks[self.CHIN].y * h),
                (landmarks[self.L_EYE].x * w, landmarks[self.L_EYE].y * h),
                (landmarks[self.R_EYE].x * w, landmarks[self.R_EYE].y * h),
                (landmarks[self.L_MOUTH].x * w, landmarks[self.L_MOUTH].y * h),
                (landmarks[self.R_MOUTH].x * w, landmarks[self.R_MOUTH].y * h)
            ], dtype=np.float64)
            
            cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
            _, rvec, _ = cv2.solvePnP(self.model_pts, pts, cam, np.zeros(4))
            rmat, _ = cv2.Rodrigues(rvec)
            
            sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
            yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
            pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
            
            return float(yaw), float(pitch)
        except:
            return 0.0, 0.0
    
    def _update_person_tracks(self, detections):
        tids = list(self.person_tracks.keys())
        
        for tid in tids:
            self.person_tracks[tid]["lost"] += 1
        
        if not detections:
            self.person_tracks = {k: v for k, v in self.person_tracks.items()
                                  if v["lost"] < self.max_lost_frames}
            return
        
        if not tids:
            for det in detections:
                self.person_tracks[self.next_person_id] = {
                    "bbox": det, "age": 1, "lost": 0,
                    "face_data": None, "face_last_seen": 999
                }
                self.next_person_id += 1
            return
        
        cost = np.zeros((len(detections), len(tids)))
        for i, det in enumerate(detections):
            for j, tid in enumerate(tids):
                cost[i, j] = 1 - self._iou(det, self.person_tracks[tid]["bbox"])
        
        rows, cols = linear_sum_assignment(cost)
        matched = set()
        
        for r, c in zip(rows, cols):
            if cost[r, c] < 0.65:
                tid = tids[c]
                self.person_tracks[tid]["bbox"] = detections[r]
                self.person_tracks[tid]["age"] += 1
                self.person_tracks[tid]["lost"] = 0
                matched.add(r)
        
        for i, det in enumerate(detections):
            if i not in matched:
                self.person_tracks[self.next_person_id] = {
                    "bbox": det, "age": 1, "lost": 0,
                    "face_data": None, "face_last_seen": 999
                }
                self.next_person_id += 1
        
        self.person_tracks = {k: v for k, v in self.person_tracks.items()
                              if v["lost"] < self.max_lost_frames}
    
    def detect(self, frame) -> Dict[int, dict]:
        h, w = frame.shape[:2]
        
        res = self.yolo(frame, imgsz=1280, conf=CONFIG["detection_confidence"],
                        classes=[0], verbose=False)[0]
        
        person_detections = []
        if res.boxes is not None:
            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                person_detections.append(tuple(coords))
        
        self._update_person_tracks(person_detections)
        
        result = {}
        
        for tid, track in self.person_tracks.items():
            if track["lost"] > 0:
                if track.get("face_data") and track["face_last_seen"] < FPS * 0.5:
                    result[tid] = track["face_data"].copy()
                    result[tid]["is_carried"] = True
                continue
            
            head_roi = self._get_head_roi(track["bbox"], frame.shape)
            rx1, ry1, rx2, ry2 = head_roi
            
            if ry2 <= ry1 or rx2 <= rx1:
                continue
            
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue
            
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            face_result = self.face_mesh.process(rgb)
            
            if face_result.multi_face_landmarks:
                landmarks = face_result.multi_face_landmarks[0].landmark
                bbox = self._get_face_box(landmarks, frame.shape, head_roi)
                
                if bbox[2] - bbox[0] < CONFIG["min_face_size"]:
                    continue
                
                yaw, pitch = self._get_head_pose(landmarks, roi.shape)
                
                face_data = {"bbox": bbox, "yaw": yaw, "pitch": pitch, "is_carried": False}
                track["face_data"] = face_data
                track["face_last_seen"] = 0
                result[tid] = face_data
            else:
                track["face_last_seen"] = track.get("face_last_seen", 0) + 1
                if track.get("face_data") and track["face_last_seen"] < FPS * 0.5:
                    result[tid] = track["face_data"].copy()
                    result[tid]["is_carried"] = True
        
        return result
    
    def get_lost_tracks(self) -> Dict[int, int]:
        carry_frames = int(CONFIG["face_missing_hold_seconds"] * FPS)
        return {tid: track["lost"] for tid, track in self.person_tracks.items()
                if 0 < track["lost"] <= carry_frames and track.get("face_data")}


# ============================================================================
# ATTENTION ANALYZER - CONSERVATIVE
# ============================================================================

class AttentionAnalyzer:
    """
    Conservative analyzer with BOARD-RELATIVE yaw.
    
    States:
     FOCUSED: Normal posture facing board (|board_yaw| ≤ 25°, upright pitch)
     TAKING_NOTES: Looking down at desk (|board_yaw| ≤ 25°, pitch 18-50°)
     LOOKING_AWAY: Sideways from board (|board_yaw| > 25°, transient <2s)
     DISTRACTED: LOOKING_AWAY that persists >2s (rare)
    """
    
    def __init__(self, calibrator: BoardCalibrator):
        self.calibrator = calibrator
        self.crowd_prior = CrowdPrior()
        self.students: Dict[int, StudentRecord] = {}
        self.frame_interval = 1.0 / FPS
    
    def analyze(self, tid: int, raw_face_data: dict, sid: str, timestamp: float,
                is_visible: bool = True) -> StudentRecord:
        
        if tid not in self.students:
            self.students[tid] = StudentRecord(
                student_id=sid, track_id=tid, first_seen=timestamp
            )
        
        student = self.students[tid]
        student.last_seen = timestamp
        student.total_frames += 1
        
        if is_visible:
            raw_yaw = raw_face_data["yaw"]
            pitch = raw_face_data["pitch"]
            
            # Convert to BOARD-RELATIVE yaw
            board_yaw = self.calibrator.get_board_relative_yaw(raw_yaw)
            
            student.face_data = FaceData(
                bbox=raw_face_data["bbox"],
                raw_yaw=raw_yaw,
                board_yaw=board_yaw,
                pitch=pitch
            )
            
            student.pose_history.append({
                "time": timestamp,
                "yaw": board_yaw,
                "pitch": pitch
            })
            
            # Calculate stability
            self._calculate_stability(student)
            
            # Detect nodding 
            student.is_nodding = self._detect_nodding(student)
            
            # Classify state
            raw_state = self._classify_state(student, timestamp)
            
            # Apply transition rules
            final_state = self._apply_transitions(student, raw_state, timestamp)
            
            # Update state
            student.previous_state = student.current_state
            student.current_state = final_state
            
            # Track timing
            if final_state == AttentionState.FOCUSED:
                student.last_focused_time = timestamp
            elif final_state == AttentionState.TAKING_NOTES:
                student.last_notes_time = timestamp
            elif final_state == AttentionState.LOOKING_AWAY:
                if student.previous_state not in [AttentionState.LOOKING_AWAY, AttentionState.DISTRACTED]:
                    student.looking_away_start_time = timestamp
            
            # Update stats
            student.state_durations[final_state] += self.frame_interval
            student.total_visible_time += self.frame_interval
            student.last_visible_state = final_state
            student.last_visible_time = timestamp
            student.frames_not_visible = 0
            
        else:
            student.frames_not_visible += 1
            student.current_state = self._handle_not_visible(student, timestamp)
        
        student.focus_index = self._calculate_focus_index(student)
        self.crowd_prior.update(self.students, timestamp)
        
        return student
    
    def _calculate_stability(self, student: StudentRecord):
        history = list(student.pose_history)
        if len(history) < 5:
            student.pose_variance = 0.0
            student.is_stable = True
            return
        
        recent = history[-min(len(history), int(FPS)):]
        yaws = [h["yaw"] for h in recent]
        pitches = [h["pitch"] for h in recent]
        
        student.yaw_variance = float(np.std(yaws))
        student.pitch_variance = float(np.std(pitches))
        student.pose_variance = student.yaw_variance + student.pitch_variance
        student.is_stable = student.pose_variance <= CONFIG["stable_variance_max"]
    
    def _detect_nodding(self, student: StudentRecord) -> bool:
        history = list(student.pose_history)
        if len(history) < 15:
            return False
        
        recent = history[-int(FPS * 1.5):]
        pitches = [h["pitch"] for h in recent]
        
        amplitude = max(pitches) - min(pitches)
        if amplitude < CONFIG["nod_amplitude_min"]:
            return False
        
        diffs = np.diff(pitches)
        signs = np.sign(diffs)
        sign_changes = np.sum(np.abs(np.diff(signs)) == 2)
        
        times = [h["time"] for h in recent]
        duration = times[-1] - times[0] if len(times) > 1 else 1
        freq = (sign_changes / 2) / duration if duration > 0 else 0
        
        return CONFIG["nod_freq_min"] <= freq <= CONFIG["nod_freq_max"]
    
    def _get_condition_duration(self, student: StudentRecord, check_func) -> float:
        history = list(student.pose_history)
        if not history:
            return 0.0
        
        duration = 0.0
        for i in range(len(history) - 1, -1, -1):
            h = history[i]
            if check_func(h["yaw"], h["pitch"]):
                duration += self.frame_interval
            else:
                break
        return duration
    
    def _classify_state(self, student: StudentRecord, timestamp: float) -> AttentionState:
        """
        Classify based on BOARD-RELATIVE yaw and pitch.
        Includes 'Strong Focus Override' to prevent Green Starvation.
        """
        
        board_yaw = student.face_data.board_yaw
        pitch = student.face_data.pitch
        abs_yaw = abs(board_yaw)
        
        # Crowd prior bonus
        yaw_bonus = self.crowd_prior.get_yaw_bonus()
        focused_yaw_max = CONFIG["focused_yaw_max"] + yaw_bonus
        
        # Pitch ranges
        is_upright = CONFIG["upright_pitch_min"] <= pitch <= CONFIG["upright_pitch_max"]
        is_looking_down = CONFIG["notes_pitch_min"] < pitch <= CONFIG["notes_pitch_max"]
        is_drooping = pitch > CONFIG["notes_pitch_max"]

        # ====================================================================
        # 1. STRONG FOCUS OVERRIDE 
        # ====================================================================
        # If the student is looking DIRECTLY at the board (within tight bounds),
        # we IGNORE stability checks and duration timers.
        # This catches students who are typing (unstable) but listening.
        # Bounds: Yaw ±30°, Pitch -15° to +35° (generous upright range)
        if abs_yaw <= 30.0 and -15.0 <= pitch <= 35.0:
            return AttentionState.FOCUSED

        # ====================================================================
        # 2. STANDARD CLASSIFICATION (With Stability Checks)
        # ====================================================================
        
        # === FOCUSED: Facing board with upright posture ===
        if abs_yaw <= focused_yaw_max and is_upright:
            # Note: We enforce stability here for wider angles to avoid false positives
            if student.is_stable:
                return AttentionState.FOCUSED
            
            # If unstable but recently focused, give benefit of doubt
            if student.previous_state == AttentionState.FOCUSED:
                return AttentionState.FOCUSED

        # === FOCUSED with nodding boost ===
        if student.is_nodding and abs_yaw <= focused_yaw_max:
            return AttentionState.FOCUSED
        
        # === TAKING NOTES: Facing board but looking down ===
        if abs_yaw <= focused_yaw_max and is_looking_down:
            # Less strict on stability for note-taking (writing involves movement)
            if student.pose_variance < (CONFIG["stable_variance_max"] * 1.5):
                return AttentionState.TAKING_NOTES
        
        # === LOOKING AWAY: Sideways from board ===
        if abs_yaw > CONFIG["looking_away_yaw_min"]:
            return AttentionState.LOOKING_AWAY
        
        # === DROOPING: Head drooping (might be sleepy) ===
        if is_drooping:
            return AttentionState.LOOKING_AWAY
        
        # === DEFAULT: When in doubt, lean toward previous state or Green ===
        if abs_yaw <= focused_yaw_max:
            if is_looking_down:
                return AttentionState.TAKING_NOTES
            return AttentionState.FOCUSED
        
        return AttentionState.LOOKING_AWAY
    
    def _apply_transitions(self, student: StudentRecord, 
                           new_state: AttentionState, 
                           timestamp: float) -> AttentionState:
        """
        Apply state transition rules with 'Green Stickiness' to prevent flickering.
        
        Key principles:
        1. Easy to enter Green (via _classify_state).
        2. Hard to leave Green (via Stickiness here).
        3. Hard to enter Red (via Durations here).
        """
        
        current = student.current_state
        
        if current == AttentionState.NOT_VISIBLE:
            return new_state
        
        # ====================================================================
        # 1. GREEN STICKINESS (Prevents Flickering)
        # ====================================================================
        # If currently FOCUSED/NOTES, but the raw classifier says LOOKING_AWAY,
        # we check if the violation is minor. If so, FORCE stay Green.
        if current in [AttentionState.FOCUSED, AttentionState.TAKING_NOTES] and \
           new_state == AttentionState.LOOKING_AWAY:
            
            abs_yaw = abs(student.face_data.board_yaw)
            pitch = student.face_data.pitch
           
            # If yaw is under 50° (wider than strict entry) and pitch is reasonable, stay Green.
            if abs_yaw < 50.0 and pitch < 40.0:
                 return current
            
    
            # If this "Looking Away" state just started (< 0.3s), ignore it.
            # This filters out detection noise or rapid head flicks.
            time_since_last_green = timestamp - max(student.last_focused_time, student.last_notes_time)
            if time_since_last_green < 0.3:
                return current

        # ====================================================================
        # 2. QUICK RECOVERY 
        # ====================================================================
        if new_state in [AttentionState.FOCUSED, AttentionState.TAKING_NOTES]:
            return new_state
        
        # ====================================================================
        # 3. DISTRACTION LOGIC (Yellow -> Red)
        # ====================================================================
        if current in [AttentionState.LOOKING_AWAY, AttentionState.DISTRACTED]:
            if new_state == AttentionState.LOOKING_AWAY:
                # Check duration
                looking_duration = timestamp - student.looking_away_start_time
                
                # Base threshold
                duration_threshold = CONFIG["looking_away_max_duration"]
                
                # Bonus from crowd (if everyone else is paying attention, be lenient)
                duration_threshold += self.crowd_prior.get_duration_bonus()
                
                # Extreme yaw penalty (looking behind = faster distraction)
                abs_yaw = abs(student.face_data.board_yaw)
                if abs_yaw > CONFIG["looking_away_yaw_max"]:
                    duration_threshold = CONFIG["extreme_yaw_fast_track"]
                
                # Grace period: If recently focused, block Red state
                time_since_focused = timestamp - max(student.last_focused_time, student.last_notes_time)
                if time_since_focused < CONFIG["recovery_grace_period"]:
                    return AttentionState.LOOKING_AWAY
                
                # Stability check: If moving (talking/fidgeting), block Red state
            
                if not student.is_stable:
                    return AttentionState.LOOKING_AWAY
                
                # Final Trigger
                if looking_duration >= duration_threshold:
                    return AttentionState.DISTRACTED
                
                return AttentionState.LOOKING_AWAY
        
        # Keep DISTRACTED if still meeting criteria
        if current == AttentionState.DISTRACTED and new_state == AttentionState.LOOKING_AWAY:
            return AttentionState.DISTRACTED
        
        return new_state
    
    def _handle_not_visible(self, student: StudentRecord, timestamp: float) -> AttentionState:
        time_not_visible = student.frames_not_visible * self.frame_interval
        
        if time_not_visible < CONFIG["face_missing_hold_seconds"]:
            return student.last_visible_state
        
        # After hold period → LOOKING_AWAY (never DISTRACTED)
        return AttentionState.LOOKING_AWAY
    
    def _calculate_focus_index(self, student: StudentRecord) -> float:
        if student.total_visible_time <= 0:
            return 0.0
        
        weighted_sum = 0.0
        for state, duration in student.state_durations.items():
            weight = STATE_INFO[state]["weight"]
            if weight is not None:
                weighted_sum += weight * duration
        
        return min(100, (weighted_sum / student.total_visible_time) * 100)
    
    def get_all_students(self) -> Dict[int, StudentRecord]:
        return self.students


# ============================================================================
# VISUALIZER
# ============================================================================

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw(self, frame, students, stats, calibrator):
        out = frame.copy()
        
        for student in students.values():
            self._draw_student(out, student)
        
        self._draw_stats(out, stats, calibrator)
        return out
    
    def _draw_student(self, frame, student: StudentRecord):
        x1, y1, x2, y2 = student.face_data.bbox
        state = student.current_state
        info = STATE_INFO[state]
        color = info["color"]
        
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Corners
        cl = min(12, (x2 - x1) // 4)
        for (cx, cy), (dx, dy) in [((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                    ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))]:
            cv2.line(frame, (cx, cy), (cx + cl * dx, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + cl * dy), color, 3)
        
        # Student ID
        (tw, th), _ = cv2.getTextSize(student.student_id, self.font, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, student.student_id, (x1 + 4, y1 - 4), self.font, 0.5, (0, 0, 0), 1)
        
        # State label
        label = info["display"]
        if student.is_nodding and state == AttentionState.FOCUSED:
            label = "Focused ✓"
        
        (lw, lh), _ = cv2.getTextSize(label, self.font, 0.4, 1)
        ly = y2 + lh + 6
        cv2.rectangle(frame, (x1 - 1, ly - lh - 3), (x1 + lw + 4, ly + 3), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 2, ly), self.font, 0.4, color, 1)
        
        # Debug info
        debug = f"Y:{student.face_data.board_yaw:+.0f}° P:{student.face_data.pitch:+.0f}°"
        cv2.putText(frame, debug, (x1, y2 + lh + 22), self.font, 0.28, (180, 180, 180), 1)
    
    def _draw_stats(self, frame, stats, calibrator):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (290, 220), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, 10), (290, 220), (80, 80, 80), 1)
        
        cv2.putText(frame, "ATTENTION ANALYTICS", (18, 30), self.font, 0.45, (255, 255, 255), 1)
        cv2.line(frame, (18, 38), (280, 38), (80, 80, 80), 1)
        
        y = 55
        
        if calibrator.is_calibrated:
            cv2.putText(frame, f"Board: {calibrator.board_yaw_offset:+.1f}° from camera",
                       (18, y), self.font, 0.30, (150, 150, 150), 1)
        else:
            cv2.putText(frame, "Calibrating board...", (18, y), self.font, 0.30, (0, 200, 200), 1)
        y += 18
        
        cv2.putText(frame, f"Time: {stats['time']:.1f}s | Students: {stats['total']}",
                   (18, y), self.font, 0.32, (180, 180, 180), 1)
        y += 24
        
        # State counts
        for name, count, color in [
            ("Focused", stats.get('focused', 0), CONFIG["color_green"]),
            ("Taking Notes", stats.get('taking_notes', 0), CONFIG["color_green"]),
            ("Looking Away", stats.get('looking_away', 0), CONFIG["color_yellow"]),
            ("Distracted", stats.get('distracted', 0), CONFIG["color_red"]),
        ]:
            cv2.rectangle(frame, (18, y - 8), (26, y + 2), color, -1)
            cv2.putText(frame, f"{name}: {count}", (32, y), self.font, 0.35, color, 1)
            y += 18
        
        y += 10
        
        # Focus bar
        focus_idx = stats.get('class_focus_index', 0)
        bar_w = 230
        cv2.rectangle(frame, (18, y), (18 + bar_w, y + 18), (50, 50, 50), -1)
        filled = int(bar_w * focus_idx / 100)
        bar_color = CONFIG["color_green"] if focus_idx >= 70 else CONFIG["color_yellow"] if focus_idx >= 50 else CONFIG["color_red"]
        cv2.rectangle(frame, (18, y), (18 + filled, y + 18), bar_color, -1)
        cv2.putText(frame, f"Class Focus: {focus_idx:.0f}%", (18 + bar_w // 2 - 55, y + 14),
                   self.font, 0.40, (255, 255, 255), 1)


# ============================================================================
# REPORT
# ============================================================================

class Report:
    def __init__(self):
        self.frames = []
        self.students = {}
    
    def update(self, frame_num, timestamp, students):
        counts = defaultdict(int)
        focus_indices = []
        
        for s in students.values():
            counts[s.current_state] += 1
            focus_indices.append(s.focus_index)
        
        self.frames.append({
            "Frame": frame_num,
            "Time": round(timestamp, 2),
            "Total": len(students),
            "Focused": counts[AttentionState.FOCUSED],
            "Taking Notes": counts[AttentionState.TAKING_NOTES],
            "Looking Away": counts[AttentionState.LOOKING_AWAY],
            "Distracted": counts[AttentionState.DISTRACTED],
            "Class Focus %": round(np.mean(focus_indices), 1) if focus_indices else 0
        })
        
        for tid, s in students.items():
            self.students[tid] = s
    
    def generate(self, path, info, calibrator):
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            focus_indices = [s.focus_index for s in self.students.values()]
            class_focus = np.mean(focus_indices) if focus_indices else 0
            
            summary = [
                ["CLASSROOM ATTENTION REPORT", ""],
                ["", ""],
                ["Video", info.get("path", "N/A")],
                ["Duration (s)", f"{info.get('duration', 0):.1f}"],
                ["Students", len(self.students)],
                ["", ""],
                ["Board Offset", f"{calibrator.board_yaw_offset:+.1f}°"],
                ["", ""],
                ["STATE DEFINITIONS", ""],
                ["🟢 Focused", "Facing board, upright posture"],
                ["🟢 Taking Notes", "Facing board, looking down at desk"],
                ["🟡 Looking Away", "Sideways from board (transient)"],
                ["🔴 Distracted", "Prolonged looking away (>2s)"],
                ["", ""],
                ["CLASS FOCUS INDEX", f"{class_focus:.1f}%"],
            ]
            pd.DataFrame(summary, columns=["Metric", "Value"]).to_excel(
                writer, "Summary", index=False, header=False
            )
            
            rows = []
            for tid in sorted(self.students.keys()):
                s = self.students[tid]
                total = sum(s.state_durations.values())
                if total == 0:
                    continue
                
                rows.append({
                    "Student ID": s.student_id,
                    "Focus Index": round(s.focus_index, 1),
                    "Visible (s)": round(s.total_visible_time, 1),
                    "Focused %": round(100 * s.state_durations.get(AttentionState.FOCUSED, 0) / total, 1),
                    "Taking Notes %": round(100 * s.state_durations.get(AttentionState.TAKING_NOTES, 0) / total, 1),
                    "Looking Away %": round(100 * s.state_durations.get(AttentionState.LOOKING_AWAY, 0) / total, 1),
                    "Distracted %": round(100 * s.state_durations.get(AttentionState.DISTRACTED, 0) / total, 1),
                })
            
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("Focus Index", ascending=False)
            df.to_excel(writer, "Students", index=False)
            
            pd.DataFrame(self.frames).to_excel(writer, "Timeline", index=False)
            
            for sheet in writer.sheets:
                ws = writer.sheets[sheet]
                for col in ws.columns:
                    ml = max(len(str(c.value or "")) for c in col)
                    ws.column_dimensions[col[0].column_letter].width = min(ml + 3, 30)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global FPS
    
    print("=" * 70)
    print("  CLASSROOM ATTENTION ANALYTICS")
    print("  Conservative Philosophy - BOARD-RELATIVE")
    print("=" * 70)
    print("\n   STATES:")
    print("     🟢 FOCUSED: Normal posture facing board")
    print("     🟢 TAKING NOTES: Looking down at desk (productive)")
    print("     🟡 LOOKING AWAY: Sideways from board (transient)")
    print("     🔴 DISTRACTED: Prolonged looking away (rare)")
    print("=" * 70)
    
    video = Path(CONFIG["video_path"])
    if not video.exists():
        print(f"\n Video not found: {video}")
        sys.exit(1)
    
    print(f"\n📹 {video}")
    
    cap = cv2.VideoCapture(str(video))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(CONFIG["max_seconds"] * FPS)
    
    print(f"   {w}x{h} @ {FPS:.0f}fps | {CONFIG['max_seconds']}s")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_vid = Path(f"attention_{ts}.mp4")
    out_xls = Path(f"attention_report_{ts}.xlsx")
    
    print(f"\n📁 Output: {out_vid.name}, {out_xls.name}")
    
    tracker = FaceTracker()
    calibrator = BoardCalibrator(CONFIG["calibration_seconds"])
    analyzer = AttentionAnalyzer(calibrator)
    viz = Visualizer()
    report = Report()
    
    writer = None
    if CONFIG["output_video"]:
        writer = cv2.VideoWriter(str(out_vid), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    
    print(f"\n Board calibration: first {CONFIG['calibration_seconds']}s")
    print("-" * 70)
    
    frame_num = 0
    last_raw_faces = {}
    
    try:
        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            t = frame_num / FPS
            
            if frame_num % 3 == 1:
                raw_faces = tracker.detect(frame)
                last_raw_faces = raw_faces
            else:
                raw_faces = last_raw_faces
            
            # Calibration
            if not calibrator.is_calibrated:
                for tid, face_data in raw_faces.items():
                    variance = analyzer.students[tid].pose_variance if tid in analyzer.students else 0.0
                    calibrator.add_sample(face_data["yaw"], variance, t)
                
                if calibrator.should_calibrate(t):
                    calibrator.calibrate()
            
            # Analyze
            students = {}
            for tid, face_data in raw_faces.items():
                sid = tracker.get_student_id(tid)
                students[tid] = analyzer.analyze(tid, face_data, sid, t, is_visible=True)
            
            for tid in tracker.get_lost_tracks():
                if tid not in students and tid in analyzer.students:
                    sid = tracker.get_student_id(tid)
                    last_face = {
                        "bbox": analyzer.students[tid].face_data.bbox,
                        "yaw": analyzer.students[tid].face_data.raw_yaw,
                        "pitch": analyzer.students[tid].face_data.pitch
                    }
                    students[tid] = analyzer.analyze(tid, last_face, sid, t, is_visible=False)
            
            # Stats
            counts = defaultdict(int)
            focus_indices = []
            for s in students.values():
                counts[s.current_state] += 1
                focus_indices.append(s.focus_index)
            
            stats = {
                "time": t,
                "total": len(students),
                "focused": counts[AttentionState.FOCUSED],
                "taking_notes": counts[AttentionState.TAKING_NOTES],
                "looking_away": counts[AttentionState.LOOKING_AWAY],
                "distracted": counts[AttentionState.DISTRACTED],
                "class_focus_index": np.mean(focus_indices) if focus_indices else 0
            }
            
            report.update(frame_num, t, students)
            
            out = viz.draw(frame, students, stats, calibrator)
            
            if writer:
                writer.write(out)
            
            cv2.imshow("Attention Analytics", out)
            
            if frame_num % int(FPS) == 0:
                print(f"   {100 * frame_num / max_frames:5.1f}% | {t:.1f}s | "
                      f"🟢F:{stats['focused']} 🟢N:{stats['taking_notes']} "
                      f"🟡L:{stats['looking_away']} 🔴D:{stats['distracted']} | "
                      f"Focus: {stats['class_focus_index']:.0f}%")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print("\n Generating report...")
    report.students = analyzer.get_all_students()
    report.generate(out_xls, {"path": str(video), "duration": frame_num / FPS}, calibrator)
    
    all_students = analyzer.get_all_students()
    focus_indices = [s.focus_index for s in all_students.values()]
    class_focus = np.mean(focus_indices) if focus_indices else 0
    
    total_time = sum(s.total_visible_time for s in all_students.values())
    state_times = defaultdict(float)
    for s in all_students.values():
        for state, dur in s.state_durations.items():
            state_times[state] += dur
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"\n   Board: {calibrator.board_yaw_offset:+.1f}° from camera")
    print(f"   Class Focus: {class_focus:.1f}%")
    print(f"   Students: {len(all_students)}")
    
    if total_time > 0:
        print(f"\n  Distribution:")
        for state, emoji in [
            (AttentionState.FOCUSED, "🟢"),
            (AttentionState.TAKING_NOTES, "🟢"),
            (AttentionState.LOOKING_AWAY, "🟡"),
            (AttentionState.DISTRACTED, "🔴"),
        ]:
            pct = 100 * state_times[state] / total_time
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {emoji} {state.value:15s}: {bar} {pct:5.1f}%")
    
    print(f"\n  📁 {out_vid}")
    print(f"  📁 {out_xls}")
    print("=" * 70 + "\n")


if __name__ == "__main__":

    main()
