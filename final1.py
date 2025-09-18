# -*- coding: utf-8 -*-
# 融合版：循迹 + 丢线探前/扫描 + 前车跟随/停车 + Marker识别拍照

import cv2
import numpy as np
import time
from queue import Empty
from robomaster import robot, vision

# ========= 蓝线参数 =========
LOWER_BLUE = np.array([100, 50, 20])
UPPER_BLUE = np.array([140, 255, 255])
KP_LINE = 0.8
BASE_SPEED = 0.2
PITCH_ANGLE = -50

# ========= 丢线处理参数 =========
SEARCH_YAW_DPS = 60.0
SEARCH_SWEEP_DEG = 120.0
FOUND_STABLE_FRAMES = 2
SEARCH_SEG_TIME = SEARCH_SWEEP_DEG / abs(SEARCH_YAW_DPS)
PROBE_FWD_TIME = 0.8
PROBE_FWD_SPEED = 0.2

# ========= 前车跟随/停车参数 =========
FOLLOW_SPEED = 0.12
SLOW_H_FRAC = 0.18
STOP_H_FRAC = 0.45
HOLD_SEC = 1.0

# ========= Marker 拍照参数 =========
CENTER_TOL = 0.05
WIDTH_THRESH_NORM = 0.15
TEAM_ID = 20

# ========= 全局状态 =========
last_box = None
last_ts = 0.0
which_detector = None
markers = []

# ---------- 工具 ----------
def _parse_boxes(msg):
    boxes = []
    try:
        items = msg if isinstance(msg, (list, tuple)) else [msg]
        for it in items:
            if isinstance(it, dict):
                x = float(it.get("x", it.get("bbox_x", 0)))
                y = float(it.get("y", it.get("bbox_y", 0)))
                w = float(it.get("w", it.get("bbox_w", 0)))
                h = float(it.get("h", it.get("bbox_h", 0)))
            elif isinstance(it, (list, tuple)) and len(it) >= 4:
                x, y, w, h = [float(v) for v in it[:4]]
            else:
                continue
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
    except Exception:
        pass
    return boxes

def h_frac_from_box(box, H):
    _, _, _, h = box
    return h if h <= 1.5 else (h / float(H))

# ---------- 回调 ----------
def vision_cb(msg):
    global last_box, last_ts
    boxes = _parse_boxes(msg)
    if boxes:
        last_box = max(boxes, key=lambda b: b[2] * b[3])
        last_ts = time.time()

def try_subscribe(ep_vision):
    global which_detector
    for name in ("robot", "car", "people"):
        try:
            ep_vision.sub_detect_info(name=name, callback=vision_cb)
            which_detector = name
            print(f"[Vision] subscribed to '{name}'")
            return True
        except Exception:
            continue
    return False

def on_detect_marker(marker_info):
    global markers
    markers.clear()
    for i in range(len(marker_info)):
        x, y, w, h, info = marker_info[i]
        try:
            marker_id = int(info)
        except:
            continue
        if 1 <= marker_id <= 5 and w >= WIDTH_THRESH_NORM:
            markers.append((x, y, w, h, marker_id))

# ---------- 基础函数 ----------
def init_robot():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision

    try:
        ep_robot.set_robot_mode(mode='chassis_lead')
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.5)
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0,
                       pitch_speed=30, yaw_speed=30).wait_for_completed()
    except Exception as e:
        print(f"Init gimbal error: {e}")

    return ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision

def detect_blue_line(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        line_center_x = x + w // 2
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img, (line_center_x, y + h // 2), 5, (0, 0, 255), -1)
        return line_center_x, mask
    return None, mask

def line_following_control(line_center_x, image_width):
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 2)
    z_speed = turn * 50
    return BASE_SPEED, z_speed

# ---------- 主程序 ----------
if __name__ == '__main__':
    ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision = init_robot()
    ep_camera.start_video_stream(display=False)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    try_subscribe(ep_vision)

    searching, probing = False, False
    sweep_dir, seg_end_ts, found_cnt = +1, 0.0, 0
    probe_end_ts = 0.0

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                continue
            H, W = img.shape[:2]
            line_center, mask = detect_blue_line(img)

            # ===== Marker 检测拍照逻辑 =====
            if markers:
                mk0 = max(markers, key=lambda m: m[2])
                center_error = mk0[0] - 0.5
                if abs(center_error) < CENTER_TOL and mk0[2] >= WIDTH_THRESH_NORM:
                    print("Marker 居中，拍照中...")
                    cv2.imwrite(f"captured_marker_{mk0[4]}.jpg", img)
                    break

            # ===== 循线逻辑 =====
            if line_center is not None:
                if searching or probing:
                    found_cnt += 1
                    if found_cnt < FOUND_STABLE_FRAMES:
                        if searching:
                            ep_chassis.drive_speed(x=0, y=0,
                                z=sweep_dir*SEARCH_YAW_DPS, timeout=0.1)
                        elif probing:
                            ep_chassis.drive_speed(x=PROBE_FWD_SPEED, y=0,
                                z=0, timeout=0.1)
                        continue
                    searching, probing, found_cnt = False, False, 0
                x_speed, z_speed = line_following_control(line_center, W)
            else:
                if not searching and not probing:
                    probing, probe_end_ts = True, time.time() + PROBE_FWD_TIME
                if probing:
                    if time.time() < probe_end_ts:
                        ep_chassis.drive_speed(x=PROBE_FWD_SPEED, y=0, z=0, timeout=0.1)
                    else:
                        probing, searching = False, True
                        sweep_dir = +1
                        seg_end_ts = time.time() + SEARCH_SEG_TIME
                        found_cnt = 0
                        ep_chassis.drive_speed(x=0, y=0,
                            z=sweep_dir*SEARCH_YAW_DPS, timeout=0.1)
                    continue
                elif searching:
                    if time.time() < seg_end_ts:
                        ep_chassis.drive_speed(x=0, y=0,
                            z=sweep_dir*SEARCH_YAW_DPS, timeout=0.1)
                    else:
                        sweep_dir *= -1
                        seg_end_ts = time.time() + SEARCH_SEG_TIME
                        ep_chassis.drive_speed(x=0, y=0,
                            z=sweep_dir*SEARCH_YAW_DPS, timeout=0.1)
                    continue

            # ===== 前车跟随逻辑 =====
            active = (time.time() - last_ts) <= HOLD_SEC
            box = last_box if active else None
            x_cmd = BASE_SPEED
            if box is not None:
                hf = h_frac_from_box(box, H)
                if hf >= STOP_H_FRAC:
                    x_cmd = 0.0
                elif hf >= SLOW_H_FRAC:
                    x_cmd = FOLLOW_SPEED
                x, y, w, h = box
                if h <= 1.5:
                    x, y, w, h = int(x*W), int(y*H), int(w*W), int(h*H)
                color = (0,0,255) if x_cmd==0 else ((0,255,255) if x_cmd==FOLLOW_SPEED else (0,255,0))
                cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

            ep_chassis.drive_speed(x=x_cmd, y=0, z=-z_speed if line_center else 0, timeout=0.1)

            cv2.imshow("Camera", img)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        ep_chassis.drive_speed(x=0,y=0,z=0)
        try: ep_gimbal.recenter().wait_for_completed(timeout=2)
        except: pass
        ep_camera.stop_video_stream()
        cv2.destroyAllWindows()
        ep_robot.close()
