# 🚁 DJI Human Detection System

Real-time GPU-accelerated human detection system for DJI drones using YOLOv11 and RTMP streaming.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-Latest-green.svg)](https://nodejs.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Features

- 🎯 **Real-time Human Detection** - YOLOv11 model with GPU acceleration
- 📡 **RTMP/RTSP Streaming** - Seamless integration with DJI drones via MediaMTX
- ⚡ **GPU Optimized** - CUDA acceleration with TF32, cuDNN benchmark mode
- 🌐 **Web Interface** - Real-time visualization with Socket.IO
- 🚀 **High Performance** - 18-31 FPS on NVIDIA RTX 4050
- 🔧 **Easy Setup** - One-command launch with PowerShell script

## 📋 System Requirements

| Component | Requirement |
|-----------|------------|
| **OS** | Windows 10/11 |
| **Python** | 3.12+ |
| **Node.js** | 16+ |
| **GPU** | NVIDIA GPU with CUDA support |
| **CUDA** | 12.4+ |
| **RAM** | 8GB+ recommended |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/KavinKishoreI/Human-detection.git
cd Human-detection
```

### 2. Download MediaMTX Server

MediaMTX is required for RTSP/RTMP streaming but not included in the repo (46 MB executable).

**Download:**
1. Visit [MediaMTX Releases](https://github.com/bluenviron/mediamtx/releases/tag/v1.15.2)
2. Download `mediamtx_v1.15.2_windows_amd64.zip`
3. Extract `mediamtx.exe` to the `rtsp/` folder

```
Human-detection/
├── rtsp/
│   ├── mediamtx.exe  ← Place here
│   └── mediamtx.yml
```

### 3. Install Python Dependencies

```bash
cd Prototype
pip install -r requirements.txt
```

**Key packages:**
- `ultralytics` - YOLOv11 framework
- `opencv-python` - Computer vision
- `torch` - PyTorch with CUDA
- `socketio-client` - Real-time communication

### 4. Install Node.js Dependencies

```bash
npm install
```

This installs:
- `express` - Web server
- `socket.io` - WebSocket communication

### 5. Run the Project

```powershell
cd ..
.\run_project.ps1
```

This PowerShell script launches all three components:
1. **MediaMTX Server** (RTSP/RTMP on ports 8554, 1935)
2. **Node.js Web Server** (Port 3000)
3. **Python YOLO Analyzer** (Real-time detection)

**Alternative - Manual Start:**

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

## 🌐 Access the System

Once running, access these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| **Web Interface** | http://localhost:3000 | Live detection visualization |
| **MediaMTX UI** | http://localhost:8888 | Stream management |
| **RTSP Stream** | rtsp://localhost:8554/drone | Video stream endpoint |

## 🎮 Usage

### For DJI Drone Streaming

1. **Configure your DJI drone** to stream RTMP to:
   ```
   rtmp://[YOUR_PC_IP]:1935/drone
   ```

2. **Find your PC's IP address:**
   ```powershell
   ipconfig
   ```
   Look for "IPv4 Address" under your active network adapter.

3. **Update drone settings** (varies by model):
   - DJI Fly app → Settings → Streaming
   - Enter RTMP URL with your PC's IP
   - Start streaming

4. **Open web interface** at http://localhost:3000 to see live detections!

### For Webcam Testing

Edit `Prototype/yolo_analyzer.py`:
```python
VIDEO_SOURCE = 0  # Use default webcam (already set)
```

### For Video File Testing

```python
VIDEO_SOURCE = "path/to/your/video.mp4"
```

## ⚙️ Configuration

### Detection Settings (`Prototype/yolo_analyzer.py`)

```python
# Video source options
VIDEO_SOURCE = 0                    # Webcam
# VIDEO_SOURCE = "rtsp://localhost:8554/drone"  # DJI via MediaMTX
# VIDEO_SOURCE = "video.mp4"        # Video file

# YOLO model (auto-downloads on first run)
YOLO_MODEL_PATH = "yolo11n.pt"     # Nano (fast, 6MB)
# Alternatives: yolo11s.pt (22MB), yolo11m.pt (50MB), yolo11l.pt (100MB)

# Performance tuning
DETECTION_INTERVAL = 0.1            # Seconds between detections
YOLO_IMAGE_SIZE = 640               # Input resolution
JPEG_QUALITY = 70                   # Web stream quality
DETECT_CLASSES = [0]                # [0] = person only
```

### MediaMTX Settings (`rtsp/mediamtx.yml`)

Already configured for DJI drones. Key settings:
- RTSP port: 8554
- RTMP port: 1935
- HTTP API: 8888

## 📊 Performance

Tested on **NVIDIA GeForce RTX 4050 Laptop GPU**:

| Metric | Value |
|--------|-------|
| **FPS** | 18-31 |
| **GPU Utilization** | 22-41% |
| **Detection Confidence** | 0.4 threshold |
| **Input Resolution** | 640×640 |
| **Model Size** | 6 MB (yolo11n) |

**Optimizations Enabled:**
- ✅ TensorFloat-32 (TF32)
- ✅ cuDNN benchmark mode
- ✅ PyTorch model compilation
- ✅ Asynchronous frame processing
- ✅ OpenCV multi-threading (4 threads)

## 📁 Project Structure

```
Human-detection/
├── Prototype/
│   ├── server.js              # Express + Socket.IO server
│   ├── yolo_analyzer.py       # Main detection script
│   ├── test_human_detection.py
│   ├── test_video_source.py
│   ├── package.json           # Node.js dependencies
│   ├── requirements.txt       # Python dependencies
│   └── public/
│       └── index.html         # Web visualization interface
├── rtsp/
│   ├── mediamtx.exe          # Download separately (46 MB)
│   ├── mediamtx.yml          # Server configuration
│   └── LICENSE
├── model/
│   └── Human_detection.pt    # Git LFS pointer (optional custom model)
├── run_project.ps1           # Launch all servers
├── monitor_gpu.ps1           # GPU monitoring script
├── .gitignore                # Excludes large files
├── .gitattributes            # Git LFS configuration
└── README.md
```

## 🔍 Monitoring

### GPU Usage

Monitor real-time GPU performance:

```powershell
.\monitor_gpu.ps1
```

Displays:
- GPU utilization %
- Memory usage
- Temperature
- Power consumption

### Web Interface

The web interface (http://localhost:3000) shows:
- 📹 Live video stream with bounding boxes
- 👥 Human count
- ⏱️ Detection FPS
- 🎯 Confidence scores

## 🐛 Troubleshooting

### MediaMTX Not Found

```
Error: mediamtx.exe not found
```
**Solution:** Download MediaMTX from the [releases page](https://github.com/bluenviron/mediamtx/releases/tag/v1.15.2) and place in `rtsp/` folder.

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
```
**Solution:** Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### RTSP Connection Failed

**Check MediaMTX is running:**
- Visit http://localhost:8888
- Should see MediaMTX web interface

**Check port availability:**
```powershell
netstat -an | findstr "8554"
```

### DJI Drone Won't Connect

1. Ensure PC and drone are on the same network
2. Check firewall isn't blocking ports 1935, 8554
3. Verify RTMP URL format: `rtmp://[PC_IP]:1935/drone`
4. Test with VLC player first: `rtsp://localhost:8554/drone`

## 🎓 Learning Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [MediaMTX Documentation](https://github.com/bluenviron/mediamtx)
- [DJI Developer](https://developer.dji.com/)
- [Socket.IO Guide](https://socket.io/docs/v4/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv11 framework
- **MediaMTX** - RTSP/RTMP server by bluenviron
- **Socket.IO** - Real-time communication
- **OpenCV** - Computer vision library
- **PyTorch** - Deep learning framework

## 📧 Contact

**GitHub:** [@KavinKishoreI](https://github.com/KavinKishoreI)  
**Repository:** [Human-detection](https://github.com/KavinKishoreI/Human-detection)

---

⭐ If you find this project useful, please consider giving it a star!

Made with ❤️ for DJI drone enthusiasts and computer vision developers

