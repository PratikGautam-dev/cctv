import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import YOLO
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize detectors
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("✅ MediaPipe initialized!")

# Load YOLO model
print("📥 Loading YOLO model...")
yolo_model = YOLO('yolov8n.pt')
print("✅ YOLO model loaded!")

class HybridPhoneDetector:
    def __init__(self):
        # Detection thresholds
        self.hand_to_face_threshold = 0.65  # 65% for gesture
        self.yolo_confidence = 0.50  # 50% for YOLO phone detection
        self.cooldown_time = 2.0  # Seconds between alerts

        # Tracking
        self.last_alert_time = 0
        self.detection_count = 0
        self.score_history = deque(maxlen=10)

        # Distance-based thresholds
        self.near_distance_threshold = 0.15
        self.far_distance_threshold = 0.25

        # YOLO class ID for cell phone (COCO dataset)
        self.phone_class_id = 67  # 'cell phone' in COCO

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance"""
        if point1 is None or point2 is None:
            return float('inf')
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def estimate_person_distance(self, pose_landmarks):
        """Estimate person distance from camera"""
        if not pose_landmarks:
            return 1.0

        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        shoulder_width = self.calculate_distance(
            (left_shoulder.x, left_shoulder.y),
            (right_shoulder.x, right_shoulder.y)
        )

        if shoulder_width > 0.25:
            return 0.7  # Close
        elif shoulder_width > 0.15:
            return 1.0  # Normal
        else:
            return 1.5  # Far

    def detect_hand_to_face(self, pose_landmarks, hand_landmarks_list):
        """Detect hand-to-face gesture"""
        if not pose_landmarks or not hand_landmarks_list:
            return False, 0.0, "none"

        nose = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_ear = pose_landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = pose_landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        distance_factor = self.estimate_person_distance(pose_landmarks)
        near_threshold = self.near_distance_threshold * distance_factor
        far_threshold = self.far_distance_threshold * distance_factor

        max_score = 0.0
        detection_type = "none"

        for hand_landmarks in hand_landmarks_list:
            hand_points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            hand_center = (
                np.mean([p[0] for p in hand_points]),
                np.mean([p[1] for p in hand_points])
            )

            # Pattern 1: Hand near ear (calling)
            dist_left_ear = self.calculate_distance(hand_center, (left_ear.x, left_ear.y))
            dist_right_ear = self.calculate_distance(hand_center, (right_ear.x, right_ear.y))

            if dist_left_ear < near_threshold or dist_right_ear < near_threshold:
                max_score = max(max_score, 0.85)
                detection_type = "calling"

            # Pattern 2: Hand in front of face (texting)
            dist_nose = self.calculate_distance(hand_center, (nose.x, nose.y))

            if (left_shoulder.x < hand_center[0] < right_shoulder.x and
                hand_center[1] < nose.y and dist_nose < far_threshold):
                max_score = max(max_score, 0.70)
                detection_type = "texting"

            # Pattern 3: Both hands together
            if len(hand_landmarks_list) == 2:
                hand2_points = [(lm.x, lm.y) for lm in hand_landmarks_list[1].landmark]
                hand2_center = (
                    np.mean([p[0] for p in hand2_points]),
                    np.mean([p[1] for p in hand2_points])
                )

                hands_distance = self.calculate_distance(hand_center, hand2_center)

                if hands_distance < 0.15 * distance_factor and dist_nose < far_threshold:
                    max_score = max(max_score, 0.90)
                    detection_type = "two_hands"

        return max_score >= self.hand_to_face_threshold, max_score, detection_type

    def detect_phone_object(self, frame, yolo_results):
        """Detect phone object using YOLO"""
        phone_detections = []

        if yolo_results and len(yolo_results) > 0:
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Check if it's a cell phone with sufficient confidence
                    if cls == self.phone_class_id and conf >= self.yolo_confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        phone_detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf
                        })

        return phone_detections

    def detect(self, frame, pose_landmarks, hand_landmarks_list, yolo_results):
        """
        Hybrid detection: Combine hand-to-face gesture + YOLO object detection
        Returns: (is_phone_detected, confidence_score, detection_method, phone_boxes)
        """
        # Method 1: Hand-to-face gesture detection
        gesture_detected, gesture_score, gesture_type = self.detect_hand_to_face(
            pose_landmarks, hand_landmarks_list
        )

        # Method 2: YOLO phone object detection
        phone_boxes = self.detect_phone_object(frame, yolo_results)
        yolo_detected = len(phone_boxes) > 0
        yolo_score = max([p['confidence'] for p in phone_boxes]) if phone_boxes else 0.0

        # Combine detections
        final_score = 0.0
        detection_method = "none"

        if gesture_detected and yolo_detected:
            # Both detected - HIGHEST confidence
            final_score = 0.95
            detection_method = f"hybrid_{gesture_type}"
        elif yolo_detected:
            # Only YOLO detected phone object
            final_score = yolo_score
            detection_method = "phone_object"
        elif gesture_detected:
            # Only gesture detected
            final_score = gesture_score * 0.8
            detection_method = f"gesture_{gesture_type}"

        # Temporal smoothing
        self.score_history.append(final_score)
        avg_score = np.mean(self.score_history)

        is_detected = avg_score >= 0.65

        return is_detected, avg_score, detection_method, phone_boxes

    def should_send_alert(self, current_time):
        """Check cooldown for alerts"""
        if current_time - self.last_alert_time >= self.cooldown_time:
            self.last_alert_time = current_time
            self.detection_count += 1
            return True
        return False