@echo off
chcp 65001 >nul
title 小智抽纸唤醒固件烧录
echo ============================================
echo   小智 · 抽纸唤醒 二开固件 一键烧录
echo   (v9.9.9-tissue bread-compact-wifi OLED128x64)
echo ============================================
echo.
echo 请确认:
echo   1. 板子已用 USB 线连到电脑
echo   2. 关闭其他占用串口的程序
echo.
pause

echo.
echo [1/2] 检查/安装烧录工具 esptool...
python -m pip install -q esptool==4.* || pip install -q esptool==4.*
if errorlevel 1 (
  echo esptool 安装失败，请先安装 Python 3.9+ 后重试
  pause & exit /b 1
)

echo.
echo [2/2] 开始烧录（进度条会自己走，全程约 1-3 分钟）...
python -m esptool --chip esp32s3 --baud 921600 write_flash 0x0 xiaozhi_tissue_9.9.9.bin
if errorlevel 1 (
  echo.
  echo 烧录失败？常见原因:
  echo   - 串口被占用(关掉串口监视器)
  echo   - 没装驱动: 设备管理器里看有没有出现 COM 口
  echo     CH343 芯片请装 CH343 驱动, CP210x 请装 CP210x 驱动
  echo   - 换根 USB 线(必须是数据线,不能是充电线)
  echo   - 烧录时按住板子上的 BOOT 键再松开
  pause & exit /b 1
)

echo.
echo ============================================
echo   烧录成功! 请拔插一次电源/USB 重启设备
echo   首次开机: 屏幕显示激活码/连网提示属正常
echo   语音唤醒词照常可用, 连抽几张纸试试新功能
echo ============================================
pause
