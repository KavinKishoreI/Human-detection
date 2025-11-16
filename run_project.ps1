# Run all three servers for the Human Detection Project
# This script launches MediaMTX, Node.js server, and Python YOLO analyzer in separate windows

$ProjectRoot = $PSScriptRoot

Write-Host "===========================================
" -ForegroundColor Cyan
Write-Host "   Human Detection System Launcher" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Check if MediaMTX exists
$MediaMTXPath = Join-Path $ProjectRoot "rtsp\mediamtx.exe"
if (-not (Test-Path $MediaMTXPath)) {
    Write-Host "WARNING: mediamtx.exe not found in rtsp\ folder!" -ForegroundColor Yellow
    Write-Host "MediaMTX is only needed for DJI drone RTMP streaming." -ForegroundColor Gray
    Write-Host "Download from: https://github.com/bluenviron/mediamtx/releases/tag/v1.15.2" -ForegroundColor Cyan
    Write-Host ""
    $Continue = Read-Host "Continue without MediaMTX? (y/n)"
    if ($Continue -ne "y") {
        exit 1
    }
    $StartMediaMTX = $false
} else {
    # Ask if user wants to start MediaMTX
    Write-Host "Start MediaMTX server? (Only needed for drone RTMP streaming)" -ForegroundColor Yellow
    $StartChoice = Read-Host "Start MediaMTX? (y/n)"
    $StartMediaMTX = ($StartChoice -eq "y")
}

# 1. Start MediaMTX Server (if needed)
if ($StartMediaMTX) {
    Write-Host "Starting MediaMTX RTSP/RTMP Server..." -ForegroundColor Green
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$ProjectRoot\rtsp'; Write-Host 'MediaMTX Server Starting...' -ForegroundColor Cyan; .\mediamtx.exe"
    ) -WindowStyle Normal
    Start-Sleep -Seconds 3
} else {
    Write-Host "Skipping MediaMTX (not required for webcam)" -ForegroundColor Gray
}

# 2. Start Node.js Web Server
Write-Host "Starting Node.js Web Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ProjectRoot\Prototype'; Write-Host 'Node.js Server Starting on http://localhost:3000' -ForegroundColor Cyan; node server.js"
) -WindowStyle Normal

Start-Sleep -Seconds 3

# 3. Start Python YOLO Analyzer
Write-Host "Starting Python YOLO Analyzer..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ProjectRoot\Prototype'; Write-Host 'Python YOLO Analyzer Starting...' -ForegroundColor Cyan; python yolo_analyzer.py"
) -WindowStyle Normal

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "All servers started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Web Interface:    http://localhost:3000" -ForegroundColor Yellow
if ($StartMediaMTX) {
    Write-Host "MediaMTX UI:      http://localhost:8888" -ForegroundColor Yellow
    Write-Host "RTSP Stream:      rtsp://localhost:8554/drone" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Press Ctrl+C in each window to stop the servers" -ForegroundColor Gray
Write-Host ""
