# DJI Human Detection System

GPU-accelerated real-time human detection system for DJI drones using YOLO and RTMP streaming.

## Features
- Real-time human detection with YOLOv11
- RTMP/RTSP streaming support via MediaMTX
- GPU acceleration with NVIDIA CUDA
- Web-based visualization with Socket.IO
- Optimized for 18-31 FPS performance on RTX 4050

## System Requirements
- Python 3.13.3
- Node.js
- NVIDIA GPU with CUDA 12.4 support
- Windows (PowerShell scripts included)

## Setup Instructions

### 1. Install Dependencies

#### Python Dependencies
```bash
cd Prototype
pip install -r requirements.txt
```

Required packages:
- ultralytics (YOLO)
- opencv-python
- torch (with CUDA support)
- numpy

#### Node.js Dependencies
```bash
cd Prototype
npm install
```

### 2. Download Required Files

⚠️ **Large files are NOT included in this repository**

You need to download these files separately:

#### A. YOLO Model Files
- `model/Human_detection.pt` - Custom trained human detection model
- `pothole_training1/best.pt` - Alternative model weights

**Options:**
1. Download from [Google Drive/Dropbox link]
2. Train your own model using YOLOv11
3. Use pretrained YOLO models from Ultralytics

#### B. MediaMTX RTSP Server
- `rtsp/mediamtx.exe` - RTSP/RTMP server executable

**Download:** https://github.com/bluenviron/mediamtx/releases/tag/v1.15.2

Extract `mediamtx.exe` to the `rtsp/` folder.

### 3. Running the System

Use the provided PowerShell script to launch all components:

```powershell
.\run_project.ps1
```

This will start:
1. MediaMTX RTSP/RTMP server (port 8554, 1935)
2. Node.js web server (port 3000)
3. Python YOLO analyzer

Or start components individually:

```bash
# Terminal 1 - MediaMTX
cd rtsp
.\mediamtx.exe

# Terminal 2 - Web Server
cd Prototype
node server.js

# Terminal 3 - YOLO Analyzer
cd Prototype
python yolo_analyzer.py
```

## Configuration

### YOLO Settings (`yolo_analyzer.py`)
```python
RTSP_URL = "rtsp://localhost:8554/drone"
DETECTION_INTERVAL = 0.1  # seconds between detections
YOLO_IMAGE_SIZE = 640
JPEG_QUALITY = 70
```

### Performance Monitoring
Monitor GPU usage:
```powershell
.\monitor_gpu.ps1
```

## Project Structure
```
.
├── model/
│   └── Human_detection.pt (download required)
├── pothole_training1/
│   └── best.pt (download required)
├── Prototype/
│   ├── server.js              # Web server
│   ├── yolo_analyzer.py       # Main detection script
│   ├── test_human_detection.py
│   ├── test_video_source.py
│   └── public/
│       └── index.html         # Web interface
├── rtsp/
│   ├── mediamtx.exe (download required)
│   └── mediamtx.yml           # Server configuration
├── run_project.ps1            # Launch all servers
└── monitor_gpu.ps1            # GPU monitoring

```

## DJI Drone Setup

See [DJI_STREAMING_SETUP.md](DJI_STREAMING_SETUP.md) for detailed instructions on:
- Configuring DJI drone RTMP streaming
- Network setup
- Streaming parameters

## Performance
- **FPS:** 18-31 FPS (with detection)
- **GPU Utilization:** 22-41% (RTX 4050)
- **Detection Confidence:** 0.4 threshold
- **Resolution:** 640x640 (YOLO input)

## Optimization Features
- TensorFloat-32 (TF32) enabled
- cuDNN benchmark mode
- PyTorch model compilation (`torch.compile`)
- Asynchronous frame processing
- OpenCV multi-threading (4 threads)

## Troubleshooting

### Large File Issues
If you cloned this repo before the cleanup, you may have large files in your Git history:
```bash
git fetch origin
git reset --hard origin/main
```

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

### RTSP Connection Failed
- Check MediaMTX is running: `http://localhost:8888`
- Verify port 8554 is not blocked
- Check DJI drone IP configuration

## License
[Specify your license]

## Credits
- YOLO: Ultralytics
- MediaMTX: bluenviron
- Socket.IO: socketio.org
