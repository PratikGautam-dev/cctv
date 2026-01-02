class Config:
    # Velocity thresholds
    VELOCITY_THRESHOLD_HIGH = 0.15
    VELOCITY_THRESHOLD_MED = 0.08
    # Proximity threshold
    PROXIMITY_THRESHOLD = 0.3
    # Temporal settings
    TEMPORAL_WINDOW = 15
    # Fight detection threshold
    FIGHT_THRESHOLD = 0.65
    # Alert cooldown (seconds)
    ALERT_COOLDOWN = 5
    # Pose confidence threshold
    MIN_POSE_CONFIDENCE = 0.5
    # MediaPipe model complexity
    MODEL_COMPLEXITY = 1

config = Config()