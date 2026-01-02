import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import config

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=config.MODEL_COMPLEXITY,
    min_detection_confidence=config.MIN_POSE_CONFIDENCE,
    min_tracking_confidence=config.MIN_POSE_CONFIDENCE
)

print("✅ MediaPipe Pose initialized!")

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_landmark_coords(landmarks, idx):
    """Extract x, y coordinates from landmark"""
    if landmarks and idx < len(landmarks.landmark):
        lm = landmarks.landmark[idx]
        return [lm.x, lm.y, lm.visibility]
    return None

def normalize_coordinates(landmarks, frame_width, frame_height):
    """Convert normalized coordinates to pixel coordinates"""
    coords = {}
    landmark_names = {
        0: 'nose',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        23: 'left_hip', 24: 'right_hip'
    }

    for idx, name in landmark_names.items():
        coord = get_landmark_coords(landmarks, idx)
        if coord:
            coords[name] = {
                'x': coord[0],
                'y': coord[1],
                'visibility': coord[2]
            }

    return coords

class PoseFeatureExtractor:
    def __init__(self):
        self.history = deque(maxlen=config.TEMPORAL_WINDOW)

    def extract_features(self, coords, frame_num):
        """Extract fight-related features from pose coordinates"""
        features = {
            'velocity_left': 0,
            'velocity_right': 0,
            'arms_raised': False,
            'aggressive_stance': False,
            'forward_lean': False,
            'rapid_movement': False
        }

        if not coords:
            return features

        # Calculate velocities if we have history
        if len(self.history) > 0:
            prev_coords = self.history[-1]['coords']

            # Left wrist velocity
            if 'left_wrist' in coords and 'left_wrist' in prev_coords:
                curr = coords['left_wrist']
                prev = prev_coords['left_wrist']
                features['velocity_left'] = calculate_distance(
                    [curr['x'], curr['y']],
                    [prev['x'], prev['y']]
                )

            # Right wrist velocity
            if 'right_wrist' in coords and 'right_wrist' in prev_coords:
                curr = coords['right_wrist']
                prev = prev_coords['right_wrist']
                features['velocity_right'] = calculate_distance(
                    [curr['x'], curr['y']],
                    [prev['x'], prev['y']]
                )

        # Check for raised arms
        if all(k in coords for k in ['left_wrist', 'left_shoulder', 'right_wrist', 'right_shoulder']):
            left_raised = coords['left_wrist']['y'] < coords['left_shoulder']['y']
            right_raised = coords['right_wrist']['y'] < coords['right_shoulder']['y']
            features['arms_raised'] = left_raised or right_raised

        # Check for aggressive stance (wide stance)
        if all(k in coords for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            hip_width = abs(coords['left_hip']['x'] - coords['right_hip']['x'])
            shoulder_width = abs(coords['left_shoulder']['x'] - coords['right_shoulder']['x'])
            features['aggressive_stance'] = hip_width > (shoulder_width * 1.3)

        # Check for forward lean
        if all(k in coords for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            shoulder_y = (coords['left_shoulder']['y'] + coords['right_shoulder']['y']) / 2
            hip_y = (coords['left_hip']['y'] + coords['right_hip']['y']) / 2
            features['forward_lean'] = shoulder_y > (hip_y + 0.05)

        # Check for rapid movement
        max_velocity = max(features['velocity_left'], features['velocity_right'])
        features['rapid_movement'] = max_velocity > config.VELOCITY_THRESHOLD_HIGH

        # Store in history
        self.history.append({
            'frame': frame_num,
            'coords': coords,
            'features': features
        })

        return features

class FightDetector:
    def __init__(self):
        self.feature_extractor = PoseFeatureExtractor()
        self.last_alert_time = 0
        self.detection_count = 0

    def calculate_fight_score(self, features):
        """Calculate overall fight probability score"""
        score = 0.0

        # Velocity score (35% weight)
        max_velocity = max(features['velocity_left'], features['velocity_right'])
        if max_velocity > config.VELOCITY_THRESHOLD_HIGH:
            velocity_score = 1.0
        elif max_velocity > config.VELOCITY_THRESHOLD_MED:
            velocity_score = 0.6
        else:
            velocity_score = 0.2

        score += velocity_score * 0.35

        # Pose score (30% weight)
        pose_score = 0.0
        if features['arms_raised']:
            pose_score += 0.4
        if features['aggressive_stance']:
            pose_score += 0.3
        if features['forward_lean']:
            pose_score += 0.3

        score += pose_score * 0.30

        # Rapid movement score (20% weight)
        if features['rapid_movement']:
            score += 0.20

        # Temporal consistency (15% weight)
        if len(self.feature_extractor.history) >= 5:
            recent_scores = [
                1 if h['features']['rapid_movement'] or h['features']['arms_raised'] else 0
                for h in list(self.feature_extractor.history)[-5:]
            ]
            temporal_score = sum(recent_scores) / len(recent_scores)
            score += temporal_score * 0.15

        return min(score, 1.0)

    def detect(self, coords, frame_num, current_time):
        """Main detection function"""
        # Extract features
        features = self.feature_extractor.extract_features(coords, frame_num)

        # Calculate fight score
        fight_score = self.calculate_fight_score(features)

        # Determine if fight is detected
        is_fight = fight_score > config.FIGHT_THRESHOLD

        # Check alert cooldown
        can_alert = (current_time - self.last_alert_time) > config.ALERT_COOLDOWN

        if is_fight and can_alert:
            self.last_alert_time = current_time
            self.detection_count += 1
            return True, fight_score, features

        return False, fight_score, features