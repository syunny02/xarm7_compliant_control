@echo off
chcp 65001 >nul
title RouteB 看门狗
echo ========================================
echo  RouteB 自主代理启动中...
echo ========================================
echo.
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe E:\xarm7_door_ros2\xarm7_compliant_control\loop.py
pause
