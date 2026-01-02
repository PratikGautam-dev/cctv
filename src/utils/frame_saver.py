import cv2
import os

def send_alert(frame, timestamp, score, detection_count, output_dir="data/alerts"):
    """Send alert notification and save frame"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("🚨 FIGHT DETECTED! 🚨")
    print("="*50)
    print(f"Timestamp: {timestamp}")
    print(f"Confidence Score: {score:.2%}")
    print(f"Detection Count: {detection_count}")
    print("="*50 + "\n")

    # Save frame as evidence
    filename = os.path.join(output_dir, f"fight_detection_{detection_count}_{int(timestamp)}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Frame saved as: {filename}")

    # In a real system, you would send email/SMS here
    # send_email(frame, timestamp, score)
    # send_telegram_notification(filename, score)