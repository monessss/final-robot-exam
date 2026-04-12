#此程序是竞速，同时具有直角转弯的功能

import cv2
import numpy as np
import robomaster
from robomaster import robot
import time

# 蓝色阈值（可根据环境调整）
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

# 控制参数
KP_LINE = 0.8
BASE_SPEED = 1.3
PITCH_ANGLE = -50

# >>> 丢线扫描参数（新增）
SEARCH_YAW_DPS = 60.0          # 扫描角速度（度/秒），60°/s 比较稳
SEARCH_SWEEP_DEG = 90.0        # 每个方向扫 90°
FOUND_STABLE_FRAMES = 2        # 连续多少帧看到蓝线才判定“找到”
# 由 SEARCH_YAW_DPS 推出每个 90° 扫描段需要的时间：
SEARCH_SEG_TIME = SEARCH_SWEEP_DEG / abs(SEARCH_YAW_DPS)

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

    # >>> 扫描状态变量（新增）
    searching = False          # 是否处于“丢线扫描”模式
    sweep_dir = +1             # 扫描方向：+1 先向一侧，之后会来回反转
    seg_end_ts = 0.0           # 当前 90° 扫描段的结束时间戳
    found_cnt = 0              # 连续看到蓝线的计数

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
                # >>> 找到蓝线：若之前在搜索，则退出搜索并清零计数
                if searching:
                    found_cnt += 1
                    if found_cnt < FOUND_STABLE_FRAMES:
                        # 先“稳两帧”：仍然保持扫描角速度，避免误检
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)
                        if cv2.waitKey(1) == ord('q'):
                            break
                        continue
                    # 连续若干帧都看到，退出搜索模式
                    searching = False
                    found_cnt = 0
                    print("[SEARCH] 蓝线已找到，恢复巡线")

                # 正常巡线
                x_speed, z_speed = line_following_control(line_center, W)
                ep_chassis.drive_speed(x=x_speed, y=0, z=-z_speed, timeout=0.1)
                # print(f"底盘前进速度: {x_speed:.2f} m/s, 转向速度: {z_speed:.1f}°/s")

            else:
                # >>> 未检测到蓝线：进入/执行“左右各 90° 的扫描”
                if not searching:
                    searching = True
                    sweep_dir = +1               # 先朝一个方向开始
                    seg_end_ts = time.time() + SEARCH_SEG_TIME
                    found_cnt = 0
                    print("[SEARCH] 丢线 → 开始扫描（左右各 90°）")

                # 当前扫描段还没到时间：持续以固定角速度旋转（x=0，确保原地扫）
                if time.time() < seg_end_ts:
                    ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)
                else:
                    # 这段 90° 扫描完成 → 反向继续下一段
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


