import cv2
import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.phone_detector import (
    HybridPhoneDetector,
    pose_detector,
    hand_detector,
    yolo_model,
    mp_pose,
    mp_hands,
    mp_drawing,
    mp_drawing_styles
)

def send_phone_alert(frame, timestamp, score, detection_method, detection_count, output_dir="data/alerts"):
    """Send phone usage alert"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("📱 PHONE DETECTED! 📱")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"Confidence Score: {score:.2%}")
    print(f"Detection Method: {detection_method}")
    print(f"Detection Count: {detection_count}")
    print("="*60 + "\n")

    filename = os.path.join(output_dir, f"phone_detection_{detection_count}_{int(timestamp)}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Frame saved as: {filename}")

def process_video_phone_detection(video_path, max_frames=None, show_output=True):
    """Process video with hybrid phone detection"""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Video Info:")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds\n")

    detector = HybridPhoneDetector()

    frame_num = 0
    start_time = time.time()

    print("🔍 Starting hybrid phone detection...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_num >= max_frames:
            break

        frame_num += 1
        current_time = time.time()

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        yolo_results = yolo_model(frame, verbose=False)

        # Run MediaPipe detections
        pose_results = pose_detector.process(rgb_frame)
        hand_results = hand_detector.process(rgb_frame)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )

        # Hybrid phone detection
        is_phone_detected, phone_score, detection_method, phone_boxes = detector.detect(
            frame,
            pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None,
            hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else [],
            yolo_results
        )

        # Draw YOLO phone bounding boxes
        for phone_box in phone_boxes:
            x1, y1, x2, y2 = phone_box['bbox']
            conf = phone_box['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw info
        color = (0, 0, 255) if is_phone_detected else (0, 255, 0) if phone_score > 0.4 else (255, 255, 255)

        cv2.putText(frame, f"Score: {phone_score:.2%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if detection_method != "none":
            cv2.putText(frame, f"Method: {detection_method}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Send alert and display ONLY when phone detected
        if is_phone_detected and detector.should_send_alert(current_time):
            cv2.putText(frame, "PHONE DETECTED!", (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            send_phone_alert(frame, current_time, phone_score, detection_method, detector.detection_count)

            if show_output:
                print(f"📱 Displaying detection at frame {frame_num}/{total_frames}\n")
                cv2.imshow('Phone Detection - ALERT', frame)
                cv2.waitKey(2000)  # Show alert for 2 seconds

        # Print progress every 50 frames (but don't show video)
        if frame_num % 50 == 0:
            print(f"Processing... Frame {frame_num}/{total_frames} - Current Score: {phone_score:.2%}")

        # Allow user to quit with 'q' key
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("\n⚠️ User interrupted processing")
            break

    cap.release()
    cv2.destroyAllWindows()
    processing_time = time.time() - start_time

    print(f"\n✅ Processing complete!")
    print(f"   Total frames processed: {frame_num}")
    print(f"   Processing time: {processing_time:.2f} seconds")
    print(f"   Average FPS: {frame_num/processing_time:.2f}")
    print(f"   Total phone detections: {detector.detection_count}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📱 HYBRID PHONE DETECTION SYSTEM READY")
    print("="*60)
    print("\n🎯 Detection Methods:")
    print("   1. Hand-to-face gesture (calling/texting)")
    print("   2. YOLO object detection (phone on desk/in hand)")
    print("   3. Hybrid (both methods combined - highest confidence)")
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
    process_video_phone_detection(video_path, max_frames=3000, show_output=True)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)