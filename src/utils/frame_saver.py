import os
import cv2

CANDIDATE_DIR = "data/frames/candidates"

os.makedirs(CANDIDATE_DIR, exist_ok=True)

def save_candidate_frame(frame, frame_id, event):
    path = f"{CANDIDATE_DIR}/{event}_{frame_id}.jpg"
    cv2.imwrite(path, frame)
    return path
