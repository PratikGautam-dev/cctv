import time
from detectors.fight_detector import FightDetector
from detectors.phone_detector import detect_phone
from utils.video_reader import video_reader
from utils.frame_saver import save_candidate_frame
from utils.config import config
from verifier.gemini_verifier import verify_event

VIDEO_PATH = "data/input/test.mp4"

fight_detector = FightDetector()
candidate_frames = []

for frame_id, frame in video_reader(VIDEO_PATH):
    now = time.time()

    # ---- LOCAL DETECTION ----
    is_fight, fight_score, _ = fight_detector.detect(frame, frame_id, now)
    phone_score = detect_phone(frame)

    event_type = None

    if fight_score >= config.FIGHT_TRIGGER:
        event_type = "fight"
    elif phone_score >= config.PHONE_TRIGGER:
        event_type = "phone"

    # ---- SAVE CANDIDATE FRAMES ----
    if event_type:
        path = save_candidate_frame(frame, frame_id, event_type)
        candidate_frames.append(path)

    # ---- SEND TO GEMINI (3 FRAMES) ----
    if len(candidate_frames) == 3:
        print("📤 Sending to Gemini for verification...")
        result = verify_event(candidate_frames)
        print(result)

        candidate_frames.clear()
