# -*- coding: utf-8 -*-
# 线-follow + 智能识别前车 → 跟随/停车（保持原始巡线结构）

import cv2
import numpy as np
import robomaster
from robomaster import robot
import time

# ========= 你原始巡线的参数（保持不变） =========
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])
KP_LINE = 0.8                 # 循线比例系数
BASE_SPEED = 0.2              # 基础前进速度
PITCH_ANGLE = -50             # 云台低头角度（负值表示低头）

# ========= 新增：前车跟随/停车参数 =========
CRUISE_SPEED = BASE_SPEED     # 正常沿线速度（沿用你的 BASE_SPEED）
FOLLOW_SPEED = 0.12           # 检到前车但未到停车阈值 → 跟随速度
SLOW_H_FRAC  = 0.18           # 目标框高占全帧高度 ≥ 此值 → 减速
STOP_H_FRAC  = 0.45           # 目标框高占全帧高度 ≥ 此值 → 停车
HOLD_SEC     = 1.0            # 识别结果“有效期”，抖动时适当加大（0.8~1.2）

# ========= 识别缓存（由回调更新） =========
last_box = None      # (x, y, w, h) 可能是像素，也可能是归一化
last_ts  = 0.0
which_detector = None  # 实际订阅到的类别名：robot/car/people

# ---------- 工具：解析不同 SDK 的回调数据 ----------
def _parse_boxes(msg):
    """兼容 dict / list[dict] / list[tuple]，返回 [(x,y,w,h), ...]"""
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
    """自适应像素/归一化坐标：返回目标框高占比（0~1）"""
    _, _, _, h = box
    return h if h <= 1.5 else (h / float(H))

# ---------- Vision 回调 ----------
def vision_cb(msg):
    global last_box, last_ts
    boxes = _parse_boxes(msg)
    if boxes:
        # 取面积最大的目标作为“前方障碍（前车/行人等）”
        last_box = max(boxes, key=lambda b: b[2] * b[3])
        last_ts  = time.time()

def try_subscribe(ep_vision):
    """按优先级尝试订阅 robot → car → people；成功返回 True"""
    global which_detector
    for name in ("robot", "car", "people"):
        try:
            ep_vision.sub_detect_info(name=name, callback=vision_cb)
            # 某些版本需要显式开关
            try:
                if name == "robot" and hasattr(ep_vision, "robot_detection"):
                    ep_vision.robot_detection(True)
                if name == "car" and hasattr(ep_vision, "car_detection"):
                    ep_vision.car_detection(True)
                if name == "people" and hasattr(ep_vision, "people_detection"):
                    ep_vision.people_detection(True)
            except Exception:
                pass
            which_detector = name
            print(f"[Vision] subscribed to '{name}'")
            return True
        except Exception:
            continue
    print("[Vision] subscribe failed (no 'robot'/'car'/'people')")
    return False

# ========= 以下是你原始的巡线函数（保持不变） =========
def init_robot():
    """初始化机器人组件，设置底盘跟随云台，云台固定低头"""
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # 初始化核心组件
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal

    try:
        # 设置机器人模式为底盘跟随云台
        print("设置机器人模式为底盘跟随云台...")
        ep_robot.set_robot_mode(mode='chassis_lead')

        # 云台回正
        print("正在回正云台...")
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.5)

        # 云台低头看地面
        print(f"调整云台角度看向地面，pitch={PITCH_ANGLE}°...")
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.5)

        print("云台角度设置完成，开始循线...")
    except Exception as e:
        print(f"云台设置异常: {e}")

    return ep_robot, ep_camera, ep_chassis, ep_gimbal

def detect_blue_line(img):
    """检测蓝线并返回中心位置"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # 形态学处理
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        line_center_x = x + w // 2

        # 绘制检测结果
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img, (line_center_x, y + h // 2), 5, (0, 0, 255), -1)
        return line_center_x, mask

    return None, mask

def line_following_control(line_center_x, image_width):
    """计算底盘运动控制量（原公式不变）"""
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 2)  # 归一化转向量
    z_speed = turn * 50                           # 转向速度（度/秒）
    return CRUISE_SPEED, z_speed                  # 前进速度用 CRUISE_SPEED

# ========= 主程序 =========
if __name__ == '__main__':
    # 初始化机器人（底盘跟随云台，云台低头）
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()
    ep_vision = ep_robot.vision
    try_subscribe(ep_vision)  # 订阅智能识别

    # 启动相机
    ep_camera.start_video_stream(display=False)
    time.sleep(1.0)

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                continue

            H, W = img.shape[:2]
            line_center, mask = detect_blue_line(img)

            # === 先按“原始巡线”算出速度/转向 ===
            if line_center is not None:
                x_speed, z_speed = line_following_control(line_center, W)
            else:
                # 未检测到蓝线时停止
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                cv2.imshow("Camera View", img); cv2.imshow("Blue Line Mask", mask)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            # === 融合“停车/跟随”逻辑：基于智能识别最近结果 ===
            active = (time.time() - last_ts) <= HOLD_SEC
            box = last_box if active else None
            x_cmd = x_speed  # 默认按巡线速度走

            if box is not None:
                hf = h_frac_from_box(box, H)   # 框高占比
                if hf >= STOP_H_FRAC:
                    x_cmd = 0.0
                elif hf >= SLOW_H_FRAC:
                    x_cmd = FOLLOW_SPEED

                # 画框便于调参（像素/归一化都支持）
                x, y, w, h = box
                if h <= 1.5:  # 归一化 → 转像素
                    x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)
                color = (0, 0, 255) if x_cmd == 0.0 else ((0, 255, 255) if x_cmd == FOLLOW_SPEED else (0, 255, 0))
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, f"{which_detector or 'vision'} hf={hf:.2f}", (x, max(20, y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # === 下发（保持你原来 z 取负号的约定）===
            ep_chassis.drive_speed(x=x_cmd, y=0, z=-z_speed, timeout=0.1)

            # 显示图像
            cv2.imshow("Camera View", img)
            cv2.imshow("Blue Line Mask", mask)

            # 按 'q' 键退出
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # 停止底盘运动，云台回正
        ep_chassis.drive_speed(x=0, y=0, z=0)
        try:
            ep_gimbal.recenter().wait_for_completed(timeout=2)
        except:
            pass
        ep_camera.stop_video_stream()
        cv2.destroyAllWindows()
        ep_robot.close()
