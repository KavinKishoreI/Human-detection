# RTSP Connection Troubleshooting Script
param(
    [string]$IP = "10.23.143.198",
    [int]$Port = 8554,
    [string]$StreamPath = "dji"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RTSP Connection Troubleshooter" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$RTSP_URL = "rtsp://${IP}:${Port}/${StreamPath}"

# Test 1: Ping the IP
Write-Host "[1/4] Testing network connectivity..." -ForegroundColor Yellow
$ping = Test-Connection -ComputerName $IP -Count 2 -Quiet
if ($ping) {
    Write-Host "  ✓ IP $IP is reachable" -ForegroundColor Green
} else {
    Write-Host "  ✗ IP $IP is NOT reachable" -ForegroundColor Red
    Write-Host "  → Check if device is on the network" -ForegroundColor Gray
    exit 1
}

# Test 2: Check if port is open
Write-Host "[2/4] Testing RTSP port $Port..." -ForegroundColor Yellow
$tcpTest = Test-NetConnection -ComputerName $IP -Port $Port -WarningAction SilentlyContinue
if ($tcpTest.TcpTestSucceeded) {
    Write-Host "  ✓ Port $Port is OPEN" -ForegroundColor Green
} else {
    Write-Host "  ✗ Port $Port is CLOSED or FILTERED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible solutions:" -ForegroundColor Yellow
    Write-Host "  1. Start MediaMTX on the remote device:" -ForegroundColor Cyan
    Write-Host "     cd rtsp; .\mediamtx.exe" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. If MediaMTX is running, check firewall:" -ForegroundColor Cyan
    Write-Host "     New-NetFirewallRule -DisplayName 'RTSP 8554' -Direction Inbound -LocalPort 8554 -Protocol TCP -Action Allow" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. If this is a drone/camera, ensure it's streaming" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# Test 3: Check if MediaMTX binary exists locally
Write-Host "[3/4] Checking local MediaMTX..." -ForegroundColor Yellow
$LocalMediaMTX = ".\rtsp\mediamtx.exe"
if (Test-Path $LocalMediaMTX) {
    $size = (Get-Item $LocalMediaMTX).Length
    if ($size -lt 1000) {
        Write-Host "  ⚠ mediamtx.exe is a Git LFS pointer (not downloaded)" -ForegroundColor Yellow
        Write-Host "  → Download from: https://github.com/bluenviron/mediamtx/releases" -ForegroundColor Cyan
    } else {
        Write-Host "  ✓ MediaMTX binary exists locally" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ mediamtx.exe not found locally" -ForegroundColor Yellow
}

# Test 4: Try to connect with ffplay (if available)
Write-Host "[4/4] Testing RTSP stream playback..." -ForegroundColor Yellow
$ffplay = Get-Command ffplay -ErrorAction SilentlyContinue
if ($ffplay) {
    Write-Host "  Testing stream with ffplay..." -ForegroundColor Gray
    Write-Host "  Press 'q' to quit ffplay" -ForegroundColor Gray
    & ffplay -rtsp_transport tcp -timeout 5000000 $RTSP_URL
} else {
    Write-Host "  ⚠ ffplay not found (install FFmpeg to test)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Alternative test with VLC:" -ForegroundColor Cyan
    Write-Host "  1. Open VLC Media Player" -ForegroundColor Gray
    Write-Host "  2. Media → Open Network Stream" -ForegroundColor Gray
    Write-Host "  3. Paste: $RTSP_URL" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RTSP URL: $RTSP_URL" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
