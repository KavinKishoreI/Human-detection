# DJI 4K Transmission → Human Detection System

## 🚁 Complete Setup Guide for DJI Fly App Live Streaming

### Your Network Information

- **Computer IP Address**: `10.185.135.198` or `192.168.56.1`
- **RTMP Server**: `rtmp://10.185.135.198:1935/dji`
- **RTSP Output**: `rtsp://localhost:8554/dji`

---

## 📋 Prerequisites

1. ✅ DJI Drone with 4K transmission (DJI Mini 3 Pro, Air 3, Mavic 3, etc.)
2. ✅ DJI Fly App on your phone/tablet
3. ✅ Computer and phone on the **SAME WiFi network**
4. ✅ MediaMTX server running (already configured)

---

## 🔧 Setup Steps

### Step 1: Connect DJI Drone to Controller

1. Power on your DJI drone
2. Connect your phone to the DJI controller (USB-C/Lightning cable)
3. Open **DJI Fly** app
4. Wait for connection to establish

---

### Step 2: Configure Live Streaming in DJI Fly App

#### Option A: RTMP Streaming (Recommended)

1. **In DJI Fly App:**

   - Tap the **three dots (⋮)** in the top-right corner
   - Go to **Settings** → **Transmission Settings** or **Live Streaming**
   - Select **Custom RTMP**

2. **Enter Stream Details:**

   ```
   RTMP URL: rtmp://10.185.135.198:1935/dji
   Stream Key: (leave empty or use "live")
   ```

   > **Note:** Replace `10.185.135.198` with your computer's IP if different

3. **Quality Settings:**

   - Resolution: **1080p** or **720p** (4K may be too bandwidth-heavy)
   - Bitrate: **4-6 Mbps** for 1080p
   - Frame Rate: **30 FPS**

4. **Start Streaming:**
   - Tap **Start Live Streaming**
   - You should see "Live" indicator on screen

---

#### Option B: RTSP Streaming (Alternative)

Some DJI drones support direct RTSP output:

1. **Check if your drone supports RTSP:**

   - DJI Mini 3 Pro, Air 3, Mavic 3 series usually have this
   - Look for **RTSP** option in Live Streaming settings

2. **RTSP URL Format:**

   ```
   rtsp://192.168.1.1:8554/live
   ```

   (This is typically the drone's IP when connected)

3. **If RTSP is available:**
   - Note the RTSP URL from DJI Fly
   - Skip MediaMTX and connect directly to drone

---

### Step 3: Verify Stream in MediaMTX

1. **Check if stream is received:**

   ```powershell
   curl http://localhost:9997/v3/paths/list
   ```

2. **Or test with VLC/FFplay:**
   ```powershell
   ffplay rtsp://localhost:8554/dji
   ```

---

### Step 4: Update Python Analyzer to Use DJI Stream

1. **Open:** `e:\Sem-3\GeminiRoadProject\GeminiRoadProject\Prototype\yolo_analyzer.py`

2. **Change line 14:**

   ```python
   VIDEO_SOURCE = "rtsp://localhost:8554/dji"  # DJI drone stream via MediaMTX
   ```

3. **Restart Python analyzer:**
   ```powershell
   cd 'e:\Sem-3\GeminiRoadProject\GeminiRoadProject\Prototype'
   python yolo_analyzer.py
   ```

---

## 🔍 Troubleshooting

### Issue 1: "Connection Failed" in DJI Fly

**Solution:**

1. Ensure phone and computer are on **same WiFi network**
2. Check Windows Firewall:

   ```powershell
   # Allow port 1935 (RTMP)
   New-NetFirewallRule -DisplayName "MediaMTX RTMP" -Direction Inbound -Protocol TCP -LocalPort 1935 -Action Allow

   # Allow port 8554 (RTSP)
   New-NetFirewallRule -DisplayName "MediaMTX RTSP" -Direction Inbound -Protocol TCP -LocalPort 8554 -Action Allow
   ```

3. Ping your computer from phone to verify network connectivity

---

### Issue 2: Stream Lags or Stutters

**Solution:**

- Lower resolution in DJI Fly (use 720p instead of 1080p)
- Reduce bitrate to 2-3 Mbps
- Ensure strong WiFi signal
- Close other bandwidth-heavy apps

---

### Issue 3: No Video in Browser Dashboard

**Check:**

1. MediaMTX is receiving stream:

   ```powershell
   curl http://localhost:9997/v3/paths/list
   ```

2. Python analyzer is connected:

   - Look for: "Video source opened successfully: rtsp://localhost:8554/dji"

3. Restart all servers if needed

---

## 📱 Alternative: DJI Direct RTSP (No MediaMTX)

If your drone supports direct RTSP and you're connected via drone's WiFi:

1. **Find drone's RTSP URL** (usually shown in DJI Fly):

   ```
   rtsp://192.168.1.1:8554/live
   ```

2. **Update yolo_analyzer.py directly:**

   ```python
   VIDEO_SOURCE = "rtsp://192.168.1.1:8554/live"  # Direct from drone
   ```

3. **No MediaMTX needed** - connects straight to drone

---

## 🎯 Quick Start Commands

### Start All Servers:

```powershell
# Terminal 1: MediaMTX
cd 'e:\Sem-3\GeminiRoadProject\GeminiRoadProject\rtsp'
.\mediamtx.exe

# Terminal 2: Node Server
cd 'e:\Sem-3\GeminiRoadProject\GeminiRoadProject\Prototype'
node server.js

# Terminal 3: Python Analyzer (after configuring VIDEO_SOURCE)
cd 'e:\Sem-3\GeminiRoadProject\GeminiRoadProject\Prototype'
python yolo_analyzer.py
```

### Access Dashboard:

```
http://localhost:3000
```

---

## 📊 Expected Results

Once connected:

- ✅ Live 4K drone footage on dashboard
- ✅ Real-time human detection with bounding boxes
- ✅ Person counting (Small/Medium/Large by distance)
- ✅ Alert notifications when new people detected
- ✅ Cumulative tracking of unique persons

---

## 🔐 Security Notes

- This setup works on **local network only**
- For internet streaming, you'll need:
  - Port forwarding on router
  - Dynamic DNS or static IP
  - Consider using VPN for security

---

## 📞 Support

If you encounter issues:

1. Check MediaMTX logs in terminal
2. Verify network connectivity between devices
3. Test RTMP stream with VLC/FFplay first
4. Ensure DJI Fly app has latest updates

---

**Your Computer IPs:**

- `10.185.135.198` (likely your WiFi)
- `192.168.56.1` (VirtualBox network)

**Use the WiFi IP (`10.185.135.198`) for DJI streaming!**
