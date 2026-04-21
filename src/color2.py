# -*- coding: utf-8 -*-
"""
hsv_blue_tuner_s1.py
用途：使用 RoboMaster 机器人的相机做 HSV 蓝线检测/调参（带滑动条+保存阈值）
按键：
  q      退出
  s      保存阈值到 blue_hsv.json
  空格    暂停/继续（方便细调）

依赖：
  pip install robomaster opencv-python numpy
"""

import cv2
import numpy as np
import json
import os
from robomaster import robot

# ===== 默认更深蓝的初始阈值 =====
DEFAULT_LOWER = [100, 92, 30]   # H,S,V
DEFAULT_UPPER = [140, 255, 255]
SAVE_PATH = "blue_hsv.json"


def load_saved_hsv():
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("lower", DEFAULT_LOWER), data.get("upper", DEFAULT_UPPER)
        except Exception:
            pass
    return DEFAULT_LOWER, DEFAULT_UPPER


def save_hsv(lower, upper):
    data = {"lower": list(map(int, lower)), "upper": list(map(int, upper))}
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存到 {SAVE_PATH}: {data}")


def nothing(x):
    pass


def main():
    # ------- 连接机器人 -------
    # conn_type 可选 "ap"（直连热点）或 "sta"（同一Wi-Fi）
    # 你平时用哪种，就把下面的 conn_type 改成相应模式，并确保电脑和机器人网络设置正确
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="ap")  # 如需热点直连：conn_type="ap"
    except Exception as e:
        print("[ERROR] 机器人初始化失败：", e)
        return

    ep_camera = ep_robot.camera

    # 开启视频流（不弹内置窗口，由我们自己用 OpenCV 展示）
    try:
        # 可选参数：resolution='720p'/'1080p'（型号不同可能限制），fps=20/30等
        ep_camera.start_video_stream(display=False, resolution='720p')
    except Exception as e:
        print("[ERROR] 无法启动视频流：", e)
        ep_robot.close()
        return

    # ------- 创建窗口与滑动条 -------
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 480, 300)
    lower_init, upper_init = load_saved_hsv()

    cv2.createTrackbar("H_min", "Controls", lower_init[0], 179, nothing)
    cv2.createTrackbar("S_min", "Controls", lower_init[1], 255, nothing)
    cv2.createTrackbar("V_min", "Controls", lower_init[2], 255, nothing)
    cv2.createTrackbar("H_max", "Controls", upper_init[0], 179, nothing)
    cv2.createTrackbar("S_max", "Controls", upper_init[1], 255, nothing)
    cv2.createTrackbar("V_max", "Controls", upper_init[2], 255, nothing)

    paused = False
    print("[INFO] 按键：q 退出 | s 保存阈值 | 空格 暂停/继续")
    print("[TIP ] 蓝线更暗：降低 V_min；若偏白/反光：降低 S_min；偏紫/偏青：微调 H_min/H_max（蓝≈100~140）")

    try:
        while True:
            if not paused:
                # 从机器人相机读一帧（返回 OpenCV BGR 图像）
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=5)
                if frame is None:
                    print("[WARN] 读取帧超时/失败，继续重试...")
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

                frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
                hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

                # 读滑动条
                h_min = cv2.getTrackbarPos("H_min", "Controls")
                s_min = cv2.getTrackbarPos("S_min", "Controls")
                v_min = cv2.getTrackbarPos("V_min", "Controls")
                h_max = cv2.getTrackbarPos("H_max", "Controls")
                s_max = cv2.getTrackbarPos("S_max", "Controls")
                v_max = cv2.getTrackbarPos("V_max", "Controls")

                lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
                upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

                # 掩膜 & 去噪
                mask = cv2.inRange(hsv, lower, upper)
                kernel = np.ones((5, 5), np.uint8)
                mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

                result = cv2.bitwise_and(frame, frame, mask=mask_clean)
                info = f"LOWER: [{h_min:3d},{s_min:3d},{v_min:3d}]  UPPER: [{h_max:3d},{s_max:3d},{v_max:3d}]"
                cv2.putText(result, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # 显示
                cv2.imshow("Frame", frame)
                cv2.imshow("Mask", mask_clean)
                cv2.imshow("Result", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_hsv([h_min, s_min, v_min], [h_max, s_max, v_max])
            elif key == ord(' '):
                paused = not paused

    finally:
        # 关闭视频与连接
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        ep_robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
