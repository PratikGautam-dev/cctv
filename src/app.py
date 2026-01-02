import cv2
import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.fight_detector import (
    FightDetector, 
    pose_detector, 
    normalize_coordinates,
    mp_drawing,
    mp_pose,
    mp_drawing_styles
)
from utils.frame_saver import send_alert

def process_video(video_path, max_frames=None, show_output=True):
    """Process video and detect fights"""

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video Info:")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds\n")

    # Initialize detector
    detector = FightDetector()

    frame_num = 0
    start_time = time.time()

    print("🔍 Starting detection...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_num >= max_frames:
            break

        frame_num += 1
        current_time = time.time()

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose_detector.process(rgb_frame)

        # Extract coordinates
        coords = None
        if results.pose_landmarks:
            coords = normalize_coordinates(results.pose_landmarks, frame_width, frame_height)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Detect fight
        is_fight, fight_score, features = detector.detect(coords, frame_num, current_time)

        # Draw information on frame
        color = (0, 0, 255) if fight_score >= 0.75 else (0, 255, 0) if fight_score > 0.4 else (255, 255, 255)

        cv2.putText(frame, f"Score: {fight_score:.2%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Send alert and display frame ONLY if fight_score >= 0.75 (75%)
        if fight_score >= 0.75:
            cv2.putText(frame, "FIGHT DETECTED!", (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            send_alert(frame, current_time, fight_score, detector.detection_count)

            # Display ONLY the alert frame
            if show_output:
                print(f"🚨 Displaying high confidence detection at frame {frame_num}/{total_frames}\n")
                cv2.imshow('Fight Detection - ALERT', frame)
                cv2.waitKey(2000)  # Show alert for 2 seconds

        # Print progress every 50 frames (but don't show video)
        if frame_num % 50 == 0:
            print(f"Processing... Frame {frame_num}/{total_frames} - Current Score: {fight_score:.2%}")

        # Allow user to quit with 'q' key
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("\n⚠️ User interrupted processing")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    processing_time = time.time() - start_time

    print(f"\n✅ Processing complete!")
    print(f"   Total frames processed: {frame_num}")
    print(f"   Processing time: {processing_time:.2f} seconds")
    print(f"   Average FPS: {frame_num/processing_time:.2f}")
    print(f"   Total fights detected: {detector.detection_count}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FIGHT DETECTION SYSTEM - READY")
    print("="*60)
    print("\nPress 'q' to quit during processing\n")
    
    # Set video path
    video_path = "data/input/test.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        print("Please place your video file in data/input/test.mp4")
        sys.exit(1)
    
    print(f"✅ Video found: {video_path}")
    print("\n🚀 Starting processing...\n")

    # Process the video
    process_video(video_path, max_frames=3000, show_output=True)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)