# 此程序是竞速的最终版本。同时具有直角转弯的功能

import cv2
import numpy as np
import robomaster
from robomaster import robot
import time

# 蓝色阈值（可根据环境调整）
LOWER_BLUE = np.array([100, 50, 20])
UPPER_BLUE = np.array([140, 200, 200])

# 控制参数
KP_LINE = 0.8
BASE_SPEED = 0.3
PITCH_ANGLE = -50

# >>> 丢线扫描参数
SEARCH_YAW_DPS = 60.0          # 扫描角速度（度/秒）
SEARCH_SWEEP_DEG = 120.0       # 每段扫描角度（这里用120°，你原注释写“每个方向扫90°”）
FOUND_STABLE_FRAMES = 2        # 连续多少帧看到蓝线才判定“找到”
SEARCH_SEG_TIME = SEARCH_SWEEP_DEG / abs(SEARCH_YAW_DPS)

# >>> 新增：先向前探一小段（时间与速度可按场地微调）
PROBE_FWD_TIME = 0.8           # 丢线后先直行的时间（秒）
PROBE_FWD_SPEED = 0.2          # 丢线后先直行的速度（m/s）

def init_robot():
    """初始化机器人组件，设置底盘跟随云台，云台固定低头"""
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal

    try:
        print("设置机器人模式为底盘跟随云台...")
        ep_robot.set_robot_mode(mode='chassis_lead')

        print("正在回正云台...")
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.5)

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

        # 可视化
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img, (line_center_x, y + h // 2), 5, (0, 0, 255), -1)
        return line_center_x, mask

    return None, mask

def line_following_control(line_center_x, image_width):
    """计算底盘运动控制量"""
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 12)
    z_speed = turn * 123
    return BASE_SPEED, z_speed

if __name__ == '__main__':
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()
    ep_camera.start_video_stream(display=False)
    time.sleep(2)

    # 扫描与探前状态变量
    searching = False          # 是否处于“丢线扫描”模式
    sweep_dir = +1             # 扫描方向：+1/-1
    seg_end_ts = 0.0           # 当前扫描段结束时间戳
    found_cnt = 0              # 连续看到蓝线的计数

    probing = False            # >>> 新增：是否处于“先向前探一小段”阶段
    probe_end_ts = 0.0         # >>> 新增：探前结束时间戳

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                continue

            H, W = img.shape[:2]
            line_center, mask = detect_blue_line(img)

            # 可视化
            cv2.imshow("Camera View", img)
            cv2.imshow("Blue Line Mask", mask)

            if line_center is not None:
                # 找到蓝线：退出所有搜索/探前状态
                if searching or probing:
                    found_cnt += 1
                    if found_cnt < FOUND_STABLE_FRAMES:
                        # 稳两帧，避免误检；保持当前运动（若在扫描则继续角速度，若在探前则继续直行）
                        if searching:
                            ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)
                        elif probing:
                            ep_chassis.drive_speed(x=PROBE_FWD_SPEED, y=0.0, z=0.0, timeout=0.1)
                        if cv2.waitKey(1) == ord('q'):
                            break
                        continue
                    # 真正确认找到
                    searching = False
                    probing = False
                    found_cnt = 0
                    print("[SEARCH] 蓝线已找到，恢复巡线")

                # 正常巡线
                x_speed, z_speed = line_following_control(line_center, W)
                ep_chassis.drive_speed(x=x_speed, y=0, z=-z_speed, timeout=0.1)

            else:
                # 未检测到蓝线
                if not searching and not probing:
                    # >>> 新增：先向前探一小段，再进入扫描
                    probing = True
                    probe_end_ts = time.time() + PROBE_FWD_TIME
                    print(f"[SEARCH] 丢线 → 先直行 {PROBE_FWD_TIME:.2f}s，再扫描")

                if probing:
                    # 探前阶段：仅向前直行
                    if time.time() < probe_end_ts:
                        ep_chassis.drive_speed(x=PROBE_FWD_SPEED, y=0.0, z=0.0, timeout=0.1)
                    else:
                        # 探前结束 → 进入扫描
                        probing = False
                        searching = True
                        sweep_dir = +1
                        seg_end_ts = time.time() + SEARCH_SEG_TIME
                        found_cnt = 0
                        print("[SEARCH] 探前结束 → 开始左右各 90° 扫描")
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)

                elif searching:
                    # 扫描阶段：按时间片左右摆头
                    if time.time() < seg_end_ts:
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)
                    else:
                        sweep_dir *= -1
                        seg_end_ts = time.time() + SEARCH_SEG_TIME
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)

            # 退出
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        try:
            ep_gimbal.recenter().wait_for_completed(timeout=2)
        except:
            pass
        ep_camera.stop_video_stream()
        cv2.destroyAllWindows()
        ep_robot.close()
