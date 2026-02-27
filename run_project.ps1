# Run all three servers for the Human Detection Project
# This script launches MediaMTX, Node.js server, and Python YOLO analyzer in separate windows

$ProjectRoot = $PSScriptRoot

Write-Host "===========================================
" -ForegroundColor Cyan
Write-Host "   Human Detection System Launcher" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Kill any stale Node.js processes from previous runs
Write-Host "Checking for existing Node.js processes..." -ForegroundColor Gray
$ExistingNode = Get-Process -Name node -ErrorAction SilentlyContinue | Where-Object {$_.Path -like "*GeminiRoadProject*"}
if ($ExistingNode) {
    Write-Host "Found existing Node.js process(es). Stopping..." -ForegroundColor Yellow
    $ExistingNode | Stop-Process -Force
    Start-Sleep -Seconds 1
}

# Kill any stale Python processes from previous runs
$ExistingPython = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*yolo_analyzer.py*" -or $_.MainWindowTitle -like "*yolo_analyzer*"
}
if ($ExistingPython) {
    Write-Host "Found existing Python process(es). Stopping..." -ForegroundColor Yellow
    $ExistingPython | Stop-Process -Force
    Start-Sleep -Seconds 1
}

# Check if MediaMTX exists and is valid (not a Git LFS pointer)
$MediaMTXPath = Join-Path $ProjectRoot "rtsp\mediamtx.exe"
$MediaMTXValid = $false

if (-not (Test-Path $MediaMTXPath)) {
    Write-Host "WARNING: mediamtx.exe not found in rtsp\ folder!" -ForegroundColor Yellow
} else {
    # Check if it's a Git LFS pointer (small text file)
    $FileSize = (Get-Item $MediaMTXPath).Length
    if ($FileSize -lt 1000) {
        $Content = Get-Content $MediaMTXPath -Raw -ErrorAction SilentlyContinue
        if ($Content -match "git-lfs") {
            Write-Host "WARNING: mediamtx.exe is a Git LFS pointer file (not downloaded)!" -ForegroundColor Yellow
            Write-Host "To download with Git LFS:" -ForegroundColor Gray
            Write-Host "  git lfs install" -ForegroundColor Cyan
            Write-Host "  git lfs pull" -ForegroundColor Cyan
        } else {
            Write-Host "WARNING: mediamtx.exe is corrupted (only $FileSize bytes)!" -ForegroundColor Yellow
        }
    } else {
        $MediaMTXValid = $true
    }
}

if (-not $MediaMTXValid) {
    Write-Host "MediaMTX is only needed for DJI drone RTMP streaming." -ForegroundColor Gray
    Write-Host "Download manually from: https://github.com/bluenviron/mediamtx/releases/tag/v1.15.2" -ForegroundColor Cyan
    Write-Host "Extract mediamtx_v1.15.2_windows_amd64.tar.gz and place mediamtx.exe in rtsp\ folder" -ForegroundColor Cyan
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

# Select Python interpreter with project dependencies (prefer Python 3.13)
$PreferredPython = "C:\Users\kavin\AppData\Local\Programs\Python\Python313\python.exe"
if (Test-Path $PreferredPython) {
    $PythonCommand = "`"$PreferredPython`""
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py -3.13"
} else {
    $PythonCommand = "python"
}

# 3. Start Python YOLO Analyzer
Write-Host "Starting Python YOLO Analyzer..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ProjectRoot\Prototype'; Write-Host 'Python YOLO Analyzer Starting...' -ForegroundColor Cyan; $PythonCommand yolo_analyzer.py"
) -WindowStyle Normal

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "All servers started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Web Interface:    http://localhost:3000" -ForegroundColor Yellow
if ($StartMediaMTX) {
    Write-Host "MediaMTX UI:      http://localhost:8888" -ForegroundColor Yellow
    Write-Host "RTSP Stream:      rtsp://localhost:8554/dji" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Press Ctrl+C in each window to stop the servers" -ForegroundColor Gray
Write-Host ""
