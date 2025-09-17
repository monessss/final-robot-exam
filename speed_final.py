# 此版本为20小组的小车竞速最终版本
#20组在2025年9月17日的小车竞速项目中以12.59秒的成绩斩获第一名，远超第二。
#感谢所有对二十组提供帮助的人，在次，我谨代表二十组全体成员感谢你们。
import cv2
import numpy as np
import robomaster
from robomaster import robot
import time

# ===== 蓝色阈值（可根据环境调整） =====
LOWER_BLUE = np.array([100, 92,30], dtype=np.uint8)
UPPER_BLUE = np.array([140, 200, 200], dtype=np.uint8)

# ===== 控制参数 =====
KP_LINE = 0.8
BASE_SPEED = 1.3
PITCH_ANGLE = -70

# ===== 丢线扫描参数（新增）=====
SEARCH_YAW_DPS = 100.0      # 扫描角速度（度/秒）
LEFT_Z_SIGN    = -1        # “向左”旋转的 z 符号；若方向反了，改成 -1
FOUND_STABLE_FRAMES = 2    # 连续看到蓝线多少帧才算“稳定找到”

# 根据角速度换算各阶段时间
L90_TIME  = 80  / abs(SEARCH_YAW_DPS)   # 左转 90°
R180_TIME = 180.0 / abs(SEARCH_YAW_DPS)   # 右转 180°

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

        print("云台角度设置完成，开始巡线...")
    except Exception as e:
        print(f"云台设置异常: {e}")

    return ep_robot, ep_camera, ep_chassis, ep_gimbal

def detect_blue_line(img):
    """检测蓝线并返回中心位置（像素 x 坐标）与二值掩码"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # 形态学处理
    kernel = np.ones((4, 4), np.uint8)
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
    # 你原来的“除以 12”设定（偏激进），保持不变
    turn = KP_LINE * error / (image_width // 12)
    z_speed = turn * 130
    return BASE_SPEED, z_speed

if __name__ == '__main__':
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()

    ep_camera.start_video_stream(display=False)
    time.sleep(2)

    # ===== 扫描状态机（新增）=====
    searching      = False       # 是否处于丢线扫描模式
    phase          = None        # 'L90' 或 'R180'
    phase_end_ts   = 0.0         # 当前阶段结束时间
    found_cnt      = 0           # 连续检测到蓝线的帧计数

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                continue

            H, W = img.shape[:2]
            line_center, mask = detect_blue_line(img)

            # 显示窗口
            cv2.imshow("Camera View", img)
            cv2.imshow("Blue Line Mask", mask)

            # ===== 正常巡线 =====
            if line_center is not None and not searching:
                x_speed, z_speed = line_following_control(line_center, W)
                ep_chassis.drive_speed(x=x_speed, y=0, z=-z_speed, timeout=0.1)

            # ===== 丢线进入/执行扫描 =====
            if line_center is None:
                # 如果不在搜索模式，初始化“左 90°”阶段
                if not searching:
                    searching    = True
                    phase        = 'L90'
                    phase_end_ts = time.time() + L90_TIME
                    found_cnt    = 0
                    print("[SEARCH] 丢线：开始向左扫描 90°")

                # 执行当前阶段转动（只转不走）
                if phase == 'L90':
                    ep_chassis.drive_speed(x=0.0, y=0.0, z=LEFT_Z_SIGN * SEARCH_YAW_DPS, timeout=0.1)
                    if time.time() >= phase_end_ts:
                        # 左 90° 结束 → 进入右 180°
                        phase        = 'R180'
                        phase_end_ts = time.time() + R180_TIME
                        print("[SEARCH] 左 90°未找到 → 改为向右扫描 180°")
                elif phase == 'R180':
                    ep_chassis.drive_speed(x=0.0, y=0.0, z=-LEFT_Z_SIGN * SEARCH_YAW_DPS, timeout=0.1)
                    if time.time() >= phase_end_ts:
                        # 右 180° 也没找到，就循环再次左 90°
                        phase        = 'L90'
                        phase_end_ts = time.time() + L90_TIME
                        print("[SEARCH] 右 180°未找到 → 再次向左 90°（循环）")

            else:
                # —— 扫描模式下“看到线”需稳定若干帧再退出 —— #
                if searching:
                    found_cnt += 1
                    # 稳定前仍保持当前阶段的角速度（避免噪点误检）
                    if phase == 'L90':
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=LEFT_Z_SIGN * SEARCH_YAW_DPS, timeout=0.1)
                    elif phase == 'R180':
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=-LEFT_Z_SIGN * SEARCH_YAW_DPS, timeout=0.1)

                    if found_cnt >= FOUND_STABLE_FRAMES:
                        # 真的找到线 → 退出搜索，立即恢复巡线
                        searching = False
                        phase     = None
                        found_cnt = 0
                        print("[SEARCH] 蓝线已找到，恢复巡线")
                        # 直接下发一帧巡线指令更顺滑
                        x_speed, z_speed = line_following_control(line_center, W)
                        ep_chassis.drive_speed(x=x_speed, y=0, z=-z_speed, timeout=0.1)

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

