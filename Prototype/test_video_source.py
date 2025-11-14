import cv2
import sys

print("Testing video sources...")
print("-" * 50)

# Test webcam
print("\n1. Testing Webcam (index 0)...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("   ✓ Webcam 0 is available and working!")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("   ✗ Webcam 0 opened but cannot read frames")
    cap.release()
else:
    print("   ✗ Webcam 0 is not available")

# Test RTSP (quick test)
print("\n2. Testing RTSP stream (rtsp://localhost:8554/mystream)...")
print("   Note: This requires an active stream to MediaMTX")
cap = cv2.VideoCapture("rtsp://localhost:8554/mystream", cv2.CAP_FFMPEG)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("   ✓ RTSP stream is accessible and working!")
    else:
        print("   ✓ RTSP connection opened but no frames yet")
    cap.release()
else:
    print("   ✗ RTSP stream is not available (this is normal if no stream is active)")

print("\n" + "-" * 50)
print("\nRecommendations:")
print("1. If webcam works, set VIDEO_SOURCE = 0 in yolo_analyzer.py")
print("2. If using RTSP, make sure to stream to rtsp://localhost:8554/mystream first")
print("3. To use a video file, set VIDEO_SOURCE = 'path/to/video.mp4'")
print("\nTo stream from your phone to MediaMTX:")
print("  - Use an RTSP streaming app (like IP Webcam, RTSP Camera)")
print("  - Set stream address to: rtsp://<your-pc-ip>:8554/mystream")
