import cv2
import numpy as np
import robomaster
from robomaster import robot
import time

# 蓝色阈值（可根据环境调整）
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

# 控制参数
KP_LINE = 0.8 # 循线比例系数
BASE_SPEED = 1.25 # 基础前进速度
PITCH_ANGLE = -50  # 云台低头角度（负值表示低头）

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
    """计算底盘运动控制量"""
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 12)  # 归一化转向量
    z_speed = turn * 123  # 转向速度（度/秒）
    return BASE_SPEED, z_speed

if __name__ == '__main__':
    # 初始化机器人（底盘跟随云台，云台低头）
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()

    # 启动相机
    ep_camera.start_video_stream(display=False)
    time.sleep(2)

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                continue

            image_height, image_width = img.shape[:2]
            line_center, mask = detect_blue_line(img)

            # 显示图像
            cv2.imshow("Camera View", img)
            cv2.imshow("Blue Line Mask", mask)

            # 控制底盘循线
            if line_center is not None:
                x_speed, z_speed = line_following_control(line_center, image_width)
                ep_chassis.drive_speed(x=x_speed, y=0, z=-z_speed, timeout=0.1)
                print(f"底盘前进速度: {x_speed:.2f} m/s, 转向速度: {z_speed:.1f}°/s")
            else:
                # 未检测到蓝线时停止
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                print("未检测到蓝线，停止运动")

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
