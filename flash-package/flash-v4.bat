@echo off
setlocal EnableDelayedExpansion
title xiaozhi tissue-wake firmware v4 flash (keeps WiFi config)

echo ==================================================
echo   xiaozhi tissue-wake firmware v9.9.9-v4
echo   + 4 new wake words
echo   + crash fixes + safe flash (keeps WiFi config)
echo ==================================================
echo.
echo Plug in the board via USB, then
pause

echo.
echo [1/3] Installing esptool...
python -m pip install -q esptool==4.* || pip install -q esptool==4.*
if errorlevel 1 (
  echo esptool install failed. Install Python 3.9+ first.
  pause
  exit /b 1
)

echo.
echo [2/3] Flashing 5 segments (about 2-4 min)...
echo   NOTE: NVS segment is NOT touched - WiFi config preserved.
python -m esptool --chip esp32s3 --baud 921600 write_flash 0x0 bootloader.bin 0x8000 partition-table.bin 0xd000 ota_data_initial.bin 0x20000 xiaozhi.bin 0x800000 generated_assets.bin
if errorlevel 1 (
  echo.
  echo Flash failed? Common fixes:
  echo   - Close other programs using the COM port
  echo   - Install CH343 driver if no COM port appears
  echo   - Try another USB data cable
  echo   - Hold BOOT while pressing RST, then retry
  pause
  exit /b 1
)

echo.
echo ============================================
echo   Done! Unplug and replug USB to reboot.
echo   WiFi config preserved. Test wake words:
echo   - voice: ni hao xiao zhi (official)
echo   - voice: ni hao xiao chou / sheng chou / lao chou
echo   - sound: pull a few tissues continuously
echo ============================================
pause
