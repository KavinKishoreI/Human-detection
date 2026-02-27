"""
Stream webcam to MediaMTX for testing the RTSP pipeline.
This simulates a drone streaming to MediaMTX.
"""
import cv2
import subprocess
import sys
import time

# Configuration
WEBCAM_INDEX = 0
RTSP_URL = "rtsp://localhost:8554/dji"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

print("=" * 60)
print("  Webcam to MediaMTX RTSP Streamer")
print("=" * 60)
print(f"Source: Webcam {WEBCAM_INDEX}")
print(f"Target: {RTSP_URL}")
print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS} FPS")
print("=" * 60)
print()

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print(f"ERROR: Could not open webcam {WEBCAM_INDEX}")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"✓ Webcam opened: {actual_width}x{actual_height} @ {actual_fps} FPS")
print()

# FFmpeg command to stream to RTSP
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # Overwrite output
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{actual_width}x{actual_height}',
    '-r', str(FPS),
    '-i', '-',  # Input from pipe
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-b:v', '2M',
    '-maxrate', '2M',
    '-bufsize', '4M',
    '-pix_fmt', 'yuv420p',
    '-g', str(FPS),  # Keyframe interval
    '-f', 'rtsp',
    RTSP_URL
]

print("Starting FFmpeg stream to MediaMTX...")
print(f"Command: {' '.join(ffmpeg_cmd)}")
print()

try:
    # Start FFmpeg process
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("✓ Streaming started!")
    print(f"✓ Your YOLO analyzer should now connect to: {RTSP_URL}")
    print()
    print("Press Ctrl+C to stop streaming")
    print()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam")
            break
        
        try:
            # Write frame to FFmpeg stdin
            process.stdin.write(frame.tobytes())
            process.stdin.flush()
            
            frame_count += 1
            
            # Print stats every 5 seconds
            elapsed = time.time() - start_time
            if elapsed >= 5:
                fps_actual = frame_count / elapsed
                print(f"📊 Streaming: {frame_count} frames, {fps_actual:.1f} FPS")
                frame_count = 0
                start_time = time.time()
            
        except BrokenPipeError:
            print("ERROR: FFmpeg process terminated unexpectedly")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            break

except KeyboardInterrupt:
    print("\n\n⏹ Stopping stream...")
except FileNotFoundError:
    print("ERROR: FFmpeg not found!")
    print("Please install FFmpeg: https://ffmpeg.org/download.html")
    print("Or use: winget install FFmpeg")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    # Cleanup
    cap.release()
    try:
        process.stdin.close()
        process.terminate()
        process.wait(timeout=2)
    except:
        pass
    print("✓ Stream stopped")
