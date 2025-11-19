"""
Simple webcam streamer to MediaMTX using OpenCV (no FFmpeg required).
This publishes to MediaMTX which then serves it via RTSP.
Note: This uses HTTP POST to MediaMTX's WebRTC/HLS endpoints.
"""
import cv2
import time

print("=" * 60)
print("  Webcam Test for MediaMTX Pipeline")
print("=" * 60)
print()
print("SOLUTION: Your drone/camera needs to stream TO MediaMTX.")
print()
print("Current setup:")
print("  1. ✓ MediaMTX is running on localhost:8554")
print("  2. ✓ YOLO analyzer is waiting for rtsp://localhost:8554/drone")
print("  3. ✗ NO stream is being sent to MediaMTX yet")
print()
print("=" * 60)
print()
print("OPTIONS TO FIX:")
print()
print("Option 1: Use DJI Drone App")
print("-" * 60)
print("  Configure your DJI app to stream to:")
print("  RTMP: rtmp://<YOUR_PC_IP>:1935/drone")
print("  (Replace <YOUR_PC_IP> with your computer's IP)")
print()

print("Option 2: Install FFmpeg and stream webcam")
print("-" * 60)
print("  1. Install FFmpeg:")
print("     winget install FFmpeg")
print()
print("  2. Restart PowerShell, then run:")
print("     cd Prototype")
print("     python stream_to_mediamtx.py")
print()

print("Option 3: Use OBS Studio (Easiest for testing)")
print("-" * 60)
print("  1. Download OBS: https://obsproject.com/")
print("  2. Add Video Capture Device (your webcam)")
print("  3. Settings → Stream:")
print("     Service: Custom")
print("     Server: rtmp://localhost:1935/drone")
print("  4. Click 'Start Streaming'")
print()

print("Option 4: Test with sample video file")
print("-" * 60)
print("  If you have a video file, you can use VLC:")
print("  Media → Stream → Add file → Stream")
print("  Set destination to: rtsp://localhost:8554/drone")
print()

print("=" * 60)
print()

# Show system info for drone connection
import socket
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print("YOUR PC NETWORK INFO (for drone app):")
print(f"  Hostname: {hostname}")
print(f"  IP Address: {local_ip}")
print(f"  RTMP URL: rtmp://{local_ip}:1935/drone")
print(f"  RTSP URL: rtsp://{local_ip}:8554/drone")
print()
print("=" * 60)
print()

# Test if we can at least open webcam for verification
print("Testing local webcam...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"✓ Webcam working: {w}x{h}")
        print("  (But not streaming to MediaMTX yet)")
    cap.release()
else:
    print("✗ Could not open webcam")
print()

print("=" * 60)
print("WAITING FOR YOUR CHOICE...")
print("Choose one of the options above, then the analyzer will work!")
print("=" * 60)
