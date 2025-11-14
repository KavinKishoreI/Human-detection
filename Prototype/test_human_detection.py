import cv2
from ultralytics import YOLO
import time

print("Loading model...")
model = YOLO('../model/Human_detection.pt')
print(f"Model loaded. Classes: {model.names}")

print("\nOpening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Webcam opened successfully!")
print("Testing detection for 5 seconds...")
print("Stand in front of the camera...")

start_time = time.time()
frame_count = 0
detection_count = 0

while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    
    # Run detection - only for 'person' class (class 0)
    results = model(frame, conf=0.25, classes=[0], verbose=False)
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        detection_count += 1
        num_people = len(results[0].boxes)
        print(f"Frame {frame_count}: Detected {num_people} person(s)")
        
        # Draw boxes on frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Human Detection Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n--- Test Results ---")
print(f"Total frames processed: {frame_count}")
print(f"Frames with detections: {detection_count}")
print(f"Detection rate: {detection_count/frame_count*100:.1f}%")

if detection_count == 0:
    print("\n⚠️  WARNING: No people detected!")
    print("Possible issues:")
    print("  1. No one in front of the camera")
    print("  2. Camera permissions not granted")
    print("  3. Poor lighting conditions")
    print("  4. Model confidence threshold too high")
else:
    print("\n✓ Human detection is working correctly!")
