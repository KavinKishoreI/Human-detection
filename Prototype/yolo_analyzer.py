import cv2
import socketio
import base64
import numpy as np
from ultralytics import YOLO
import time
import winsound  # For Windows alert sound
import torch
from concurrent.futures import ThreadPoolExecutor
import queue

# --- GPU OPTIMIZATION SETTINGS ---
if torch.cuda.is_available():
    # Enable TF32 for faster matrix operations on Ampere GPUs (RTX 30/40 series)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN auto-tuner for optimized convolutions
    torch.backends.cudnn.benchmark = True
    # Enable cudnn for better performance
    torch.backends.cudnn.enabled = True
    # Set PyTorch to inference mode for speed
    torch.set_grad_enabled(False)
    print(f"🚀 GPU Acceleration Enabled: {torch.cuda.get_device_name(0)}")
    print(f"⚡ CUDA Cores Active, TF32 Enabled, cuDNN Optimized")
else:
    print("⚠️ GPU not available, using CPU")

# --- OpenCV OPTIMIZATION SETTINGS ---
# Enable OpenCV threading for parallel operations
cv2.setNumThreads(4)  # Use 4 threads for OpenCV operations
cv2.setUseOptimized(True)  # Enable optimized code paths

# --- CONFIGURATION ---
# VIDEO_SOURCE Options:
# - For RTSP stream: "rtsp://10.135.205.219:8554/mystream"
# - For webcam: 0 (default camera), 1 (second camera), etc.
# - For video file: "path/to/video.mp4"
# --- CONFIGURATION ---
# VIDEO_SOURCE Options:
# - For RTSP stream: "rtsp://10.135.205.219:8554/mystream"
# - For DJI Drone RTMP→RTSP: "rtsp://localhost:8554/dji"
# - For DJI Direct RTSP: "rtsp://192.168.1.1:8554/live" (drone's IP)
# - For webcam: 0 (default camera), 1 (second camera), etc.
# - For video file: "path/to/video.mp4"
VIDEO_SOURCE = 0 #"rtsp://localhost:8554/dji"  # DJI stream via MediaMTX (RTMP→RTSP conversion)
YOLO_MODEL_PATH = r"E:\Sem-3\GeminiRoadProject\GeminiRoadProject\model\Human_detection.pt" # Human detection model
SERVER_URL = 'http://localhost:3000'
DETECT_CLASSES = [0]  # Class IDs to detect: [0] = person only, None = all classes

# --- PERFORMANCE OPTIMIZATIONS (HIGH GPU UTILIZATION MODE) ---
YOLO_IMAGE_SIZE = 640        # Keep 640px resolution for accurate detection
FRAME_SKIP = 1               # Capture all frames for smooth video
DETECTION_INTERVAL = 0.1     # Run detection every 0.1 seconds (10 times per second for higher GPU load)
JPEG_QUALITY = 70            # High quality for better visuals (70 is sweet spot for speed+quality)
# Auto-detect GPU: Use '0' for first GPU, 'cpu' for CPU
DEVICE = '0' if torch.cuda.is_available() else 'cpu'  # Auto-detect GPU
# GPU batch processing settings
STREAM_BUFFER_SIZE = 1       # Process frames immediately for low latency

# ALERT CONFIG
ALERT_COOLDOWN = 10  # seconds to wait before alerting again for same detected entity

# --- SIZE CLASSIFICATION CONFIGURATION ---
SMALL_THRESHOLD = 5000     # Area < 5000 pixels = "Small" (far/child)
MEDIUM_THRESHOLD = 15000   # 5000 <= Area < 15000 pixels = "Medium" (normal distance)

# Font and Drawing Configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Color configuration for size classification (BGR format)
COLOR_MAP = {
    'Small': (0, 255, 0),      # Green (far/small)
    'Medium': (0, 165, 255),   # Orange (medium distance)
    'Large': (0, 0, 255)       # Red (close/large)
}

def get_size_category(area):
    """Classifies person size based on bounding box area (correlates with distance)."""
    if area < SMALL_THRESHOLD:
        return 'Small'
    elif area < MEDIUM_THRESHOLD:
        return 'Medium'
    else:
        return 'Large'

def draw_detections_on_frame(frame, detections_cache):
    """Draws stored detection boxes on frame for smooth transitions."""
    h_orig, w_orig = frame.shape[:2]
    target_h = int(h_orig * (YOLO_IMAGE_SIZE / w_orig))
    frame_resized = cv2.resize(frame, (YOLO_IMAGE_SIZE, target_h))
    annotated_frame = frame_resized.copy()
    overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
    
    if detections_cache:
        for detection in detections_cache:
            x1, y1, x2, y2 = detection['box']
            size_category = detection['category']
            area = detection['area']
            color = COLOR_MAP[size_category]
            text = f"Person: {size_category} ({int(area)}px)"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, FONT_THICKNESS)
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 10 if y1 > 20 else y1 + 30),
                FONT, FONT_SCALE * 0.7, color, FONT_THICKNESS - 1, cv2.LINE_AA
            )
        
        annotated_frame = cv2.addWeighted(annotated_frame, 1, overlay, 0.5, 0)
        total_count = len(detections_cache)
        
        cv2.putText(
            annotated_frame,
            f"People Detected: {total_count}",
            (10, 30),
            FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
        )
    
    return annotated_frame

def process_frame(frame, model, cumulative_counts, seen_track_ids, last_alert_times):
    """Runs YOLO detection and custom annotation/analysis on resized frame."""
    h_orig, w_orig = frame.shape[:2]
    target_h = int(h_orig * (YOLO_IMAGE_SIZE / w_orig))
    # Use optimized interpolation for faster resizing with contiguous memory
    frame_resized = cv2.resize(frame, (YOLO_IMAGE_SIZE, target_h), interpolation=cv2.INTER_LINEAR)
    # Ensure contiguous memory for faster GPU transfer
    frame_resized = np.ascontiguousarray(frame_resized)
    annotated_frame = frame_resized.copy()

    metadata = []
    total_count = 0

    try:
        results = model(
            frame_resized, 
            imgsz=YOLO_IMAGE_SIZE, 
            conf=0.4,   # Lower threshold for more detections (more GPU work)
            iou=0.65,   # Lower IOU for faster NMS
            verbose=False, 
            device=DEVICE, 
            half=True,  # FP16 for 2x speed boost on GPU
            max_det=20,  # Increase max detections for more GPU processing
            classes=DETECT_CLASSES,  # Filter for specific classes (person only)
            stream=False,  # Disable streaming for single frame processing
            agnostic_nms=True,  # Faster NMS across all classes
            retina_masks=False  # Disable mask refinement for speed
        )
    except Exception as e:
        print(f"YOLO inference error: {e}")
        return annotated_frame, 0, [], cumulative_counts, seen_track_ids, last_alert_times

    overlay = np.zeros_like(annotated_frame, dtype=np.uint8)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else None
        # Disabled mask processing for speed - use bounding boxes only
        has_masks = False
        polys = None

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box[:4])

            w = x2 - x1
            h = y2 - y1
            area = w * h  # Use box area only (faster than mask processing)

            size_category = get_size_category(area)
            color = COLOR_MAP[size_category]
            text = f"Person: {size_category} ({int(area)}px)"

            # Mask processing disabled for maximum performance
            # if has_masks and polys and i < len(polys):
            #     poly = polys[i].astype(np.int32)
            #     if len(poly) >= 3:
            #         area = cv2.contourArea(poly)
            #         size_category = get_size_category(area)
            #         color = COLOR_MAP[size_category]
            #         text = f"Person: {size_category} ({int(area)}px)"
            #         cv2.fillPoly(overlay, [poly], color)

            # Tracking and cumulative count with alert cooldown
            is_new_person = True
            track_id = None
            if track_ids is not None:
                track_id = int(track_ids[i])
                if track_id in seen_track_ids:
                    is_new_person = False
                else:
                    seen_track_ids.add(track_id)

            # Construct an alert key (prefer track_id if available, otherwise bbox signature)
            alert_key = track_id if track_id is not None else f"bbox_{x1}_{y1}_{x2}_{y2}"

            if is_new_person:
                now = time.time()
                last_time = last_alert_times.get(alert_key, 0)
                if now - last_time > ALERT_COOLDOWN:
                    # update cumulative counts and record alert time
                    cumulative_counts[size_category] += 1
                    last_alert_times[alert_key] = now

                    # ALERT: New human detected! (Silent mode for performance)
                    # Emit alert to frontend
                    sio.emit('new_person_alert', {
                        'category': size_category,
                        'timestamp': now,
                        'area': int(area)
                    })

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, FONT_THICKNESS)
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 10 if y1 > 20 else y1 + 30),
                FONT, FONT_SCALE * 0.7, color, FONT_THICKNESS - 1, cv2.LINE_AA
            )

            metadata.append({
                'box': [x1, y1, x2, y2],
                'area': int(area),
                'category': size_category
            })

        annotated_frame = cv2.addWeighted(annotated_frame, 1, overlay, 0.5, 0)
        total_count = len(metadata)

    cv2.putText(
        annotated_frame,
        f"People Detected: {total_count}",
        (10, 30),
        FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
    )

    return annotated_frame, total_count, metadata, cumulative_counts, seen_track_ids, last_alert_times

# --- SOCKET.IO SETUP ---
print(f"Connecting to Socket.IO server at {SERVER_URL}...")
sio = socketio.Client()
try:
    sio.connect(SERVER_URL, wait_timeout=5)
    print("Connected to Socket.IO server.")
except Exception as e:
    print(f"Failed to connect to Socket.IO server: {e}")
    exit()

# --- MODEL LOADING ---
print("Loading YOLO model...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    model.eval()
    
    # Optimize model for GPU inference
    if torch.cuda.is_available():
        # Note: YOLO handles device internally, no need to call model.to()
        print("✅ GPU detected and will be used automatically by YOLO")
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model.model = torch.compile(model.model, mode='max-autotune')
                print("✅ Model compiled with torch.compile for maximum speed")
            except Exception as e:
                print(f"⚠️ torch.compile not available: {e}")
        
        # Warmup GPU with dummy inference
        print("🔥 Warming up GPU...")
        dummy_input = torch.zeros((640, 640, 3), dtype=torch.uint8).numpy()
        _ = model(dummy_input, imgsz=YOLO_IMAGE_SIZE, device=DEVICE, half=True, verbose=False)
        print("🚀 GPU warmup complete")
    
    print(f"✅ YOLO model loaded and optimized. Running on device: {DEVICE}")
except Exception as e:
    print(f"ERROR: Model loading failed. Check the YOLO_MODEL_PATH: {YOLO_MODEL_PATH}")
    print(f"Detailed error: {e}")
    import traceback
    traceback.print_exc()
    sio.disconnect()
    exit()

# --- VIDEO STREAM SETUP ---
def initialize_stream():
    """Initializes the OpenCV video capture with stability settings."""
    # Check if VIDEO_SOURCE is a string (RTSP/file) or integer (webcam)
    if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.startswith('rtsp'):
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
        print(f"Attempting to connect to RTSP stream: {VIDEO_SOURCE}")
    elif isinstance(VIDEO_SOURCE, int):
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        print(f"Attempting to open webcam {VIDEO_SOURCE}...")
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        print(f"Attempting to open video file: {VIDEO_SOURCE}")
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering for low latency
        if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.startswith('rtsp'):
            cap.set(cv2.CAP_PROP_FPS, 30)  # Request higher FPS from stream
        elif isinstance(VIDEO_SOURCE, int):
            cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from webcam
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set higher resolution
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Video source opened successfully: {VIDEO_SOURCE}")
        return cap
    
    print(f"STREAM ERROR: Failed to open video source: {VIDEO_SOURCE}")
    return None

cap = initialize_stream()
if cap is None:
    sio.disconnect()
    exit()

print("Starting analysis loop.")
frame_counter = 0
emit_counter = 0  # Track frames for emission throttling
size_cumulative = {'Small': 0, 'Medium': 0, 'Large': 0}
seen_track_ids = set()
last_alert_times = {}
last_detection_time = 0  # Track when we last ran detection
last_detections = []  # Store last detection results for smooth display
last_count = 0  # Store last count
EMIT_EVERY_N_FRAMES = 1  # Emit every frame for maximum smoothness

# FPS tracking
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# Main streaming loop
while True:
    try:
        if not cap.isOpened():
            print("Stream lost, attempting to reconnect...")
            cap.release()
            time.sleep(1)
            cap = initialize_stream()
            if cap is None:
                time.sleep(2) 
                continue

        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Failed to capture frame (empty or error), retrying...")
            continue  # Skip sleep for faster retry

        frame_counter += 1
        fps_counter += 1
        current_time = time.time()
        
        # Calculate and display FPS every second
        if current_time - fps_start_time >= 1.0:
            current_fps = fps_counter / (current_time - fps_start_time)
            print(f"📊 FPS: {current_fps:.1f} | Processing: {'Detection' if (current_time - last_detection_time >= DETECTION_INTERVAL) else 'Cached'}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Run detection only if enough time has passed
        if current_time - last_detection_time >= DETECTION_INTERVAL and frame_counter % FRAME_SKIP == 0:
            # Run detection
            last_detection_time = current_time
                
            try:
                annotated_frame, count, metadata, size_cumulative, seen_track_ids, last_alert_times = process_frame(
                    frame, model, size_cumulative, seen_track_ids, last_alert_times
                )
                # Store detection results for smooth transitions
                last_detections = metadata
                last_count = count
            except Exception as e:
                print(f"FATAL PROCESSING ERROR: {e}. Using cached detections.")
                # Use cached detections if processing fails
                annotated_frame = draw_detections_on_frame(frame, last_detections)
        else:
            # Use cached detections for smooth transitions between detection intervals
            annotated_frame = draw_detections_on_frame(frame, last_detections)

        # High-quality JPEG encoding with optimized settings
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,  # Enable optimize for better compression at same quality
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0  # Disable progressive for faster encoding
        ] 
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param) 
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Emit frames at controlled rate for network efficiency
        emit_counter += 1
        if emit_counter % EMIT_EVERY_N_FRAMES == 0:
            data_to_send = {
                'frame': jpg_as_text,
                'metadata': {
                    'count': last_count,
                    'details': last_detections,
                    'cumulative': size_cumulative.copy()
                }
            }
            
            sio.emit('detection_data', data_to_send)
        
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        # Continue immediately for maximum frame rate

sio.disconnect()
cap.release()
cv2.destroyAllWindows()