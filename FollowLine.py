import time
import cv2
import numpy as np
from queue import Empty  # 读帧超时异常
from robomaster import robot

# ======= RedLight 控制器（保持不变）=======
from RedLight import RedLightController  # 你的 RedLight.py

# ---------------- 调参（保持不变） ----------------
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

KP_LINE = 0.8
BASE_SPEED = 0.2
PITCH_ANGLE = -50

# 统一显示窗口名
WINDOW_NAME = "RM View"

# 巡线开关（红灯触发时关闭；绿灯流程结束后再打开）
line_following_enabled = True

# RedLight 控制器实例（init_robot 注入硬件）
redlight_ctrl = None

# =============== 日志节流（保持不变） ===============
class RateLimiter:
    def __init__(self):
        self.last = {}
    def log(self, key: str, msg: str, min_interval: float = 0.5):
        now = time.time()
        t = self.last.get(key, 0.0)
        if now - t >= min_interval:
            print(msg)
            self.last[key] = now

rate = RateLimiter()

# ------------------- Obstacle Follow：新增（参数不破坏原巡线） -------------------
CRUISE_SPEED = BASE_SPEED     # 正常沿线速度 = 你的 BASE_SPEED
FOLLOW_SPEED = 0.12           # 检到前车但未到停车阈值 → 跟随速度
SLOW_H_FRAC  = 0.18           # 目标框高占全帧高度 ≥ 此值 → 减速
STOP_H_FRAC  = 0.45           # 目标框高占全帧高度 ≥ 此值 → 停车
HOLD_SEC     = 1.0            # 识别结果“有效期”，抖动时适当加大

last_box = None               # (x, y, w, h) 像素或归一化
last_ts  = 0.0
which_detector = None         # 实际订阅到的类别名：robot/car/people

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
    """返回目标框高占比（0~1），像素/归一化皆可"""
    _, _, _, h = box
    return h if h <= 1.5 else (h / float(H))

def vision_cb(msg):
    """更新最近目标框"""
    global last_box, last_ts
    boxes = _parse_boxes(msg)
    if boxes:
        last_box = max(boxes, key=lambda b: b[2] * b[3])  # 面积最大
        last_ts  = time.time()

def try_subscribe(ep_vision):
    """按优先级订阅 robot → car → people；成功返回 True"""
    global which_detector
    for name in ("robot", "car", "people"):
        try:
            ep_vision.sub_detect_info(name=name, callback=vision_cb)
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

# =============== 初始化（保持不变） ===============
def init_robot():
    """初始化机器人组件，设置底盘跟随云台，云台固定低头"""
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision

    try:
        # 底盘跟随云台
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

    # 注入 RedLightController（不二次连接）
    global redlight_ctrl
    redlight_ctrl = RedLightController(show_debug=False)  # 统一窗口显示，这里关闭内部窗口
    redlight_ctrl.ep = ep_robot
    redlight_ctrl.ep_gimbal = ep_gimbal
    redlight_ctrl.ep_camera = ep_camera
    redlight_ctrl.mode_state = 3
    if hasattr(redlight_ctrl, "_last_state"):
        redlight_ctrl._last_state = 3
    if hasattr(redlight_ctrl, "detect_state"):
        redlight_ctrl.detect_state = 3

    # 订阅 Obstacle Follow 的视觉流
    try_subscribe(ep_vision)

    return ep_robot, ep_camera, ep_chassis, ep_gimbal

# =============== 视觉：蓝线（保持不变） ===============
def detect_blue_line(img_bgr):
    """检测蓝线并返回中心位置；在同一张图上做可视化标注"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        line_center_x = x + w // 2

        # 在同一张图上画框 + 中心点
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img_bgr, (line_center_x, y + h // 2), 5, (0, 0, 255), -1)
        return line_center_x, mask

    return None, mask

def line_following_control(line_center_x, image_width):
    """计算底盘运动控制量"""
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 2)  # 归一化转向量
    z_speed = turn * 50  # 转向速度（度/秒）
    return BASE_SPEED, z_speed

# =============== 红灯流程（保持不变） ===============
def handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal):
    """交给 redlight_ctrl 控制；直到其 mode_state 回到 3。"""
    rate.log("rl_enter", "[红灯] 暂停巡线，进入 RedLight 控制流程...", 1.0)

    # 停车
    try:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
    except Exception:
        pass

    # 循环调用 step()，直到回到常态 3
    while True:
        det_state, mode_state, _ = redlight_ctrl.step()
        # 期间保障底盘静止
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
        except Exception:
            pass

        if mode_state == 3:
            rate.log("rl_exit", "[绿灯] RedLight 流程结束，准备恢复巡线（回正+低头）...", 1.0)
            break

    # —— 恢复巡线起点：回正 + 低头 + 模式恢复 ——
    try:
        ep_robot.set_robot_mode(mode='chassis_lead')
    except Exception:
        pass
    try:
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.3)
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.3)
    except Exception as e:
        rate.log("rl_warn_restore", f"[WARN] 恢复巡线时云台设置异常：{e}", 2.0)

# =============== 主流程（保持不变 + 融合 Obstacle Follow） ===============
if __name__ == '__main__':
    # 初始化
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()

    # 启动相机
    ep_camera.start_video_stream(display=False)
    time.sleep(1.5)

    # GUI：只用一个窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            # 取帧
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            except Empty:
                rate.log("cam_empty", "[camera] 空帧", 1.0)
                continue

            if frame is None:
                continue

            H, W = frame.shape[:2]

            # 可视化画布：统一在 vis 上叠加（红/绿 + 蓝线 + 障碍框）
            vis = frame.copy()

            # —— 红/绿检测并绘制（不触发云台动作，仅视觉） ——
            try:
                det_state = redlight_ctrl._detect_color_state(frame, draw=vis)  # 0=红,1=绿
            except Exception:
                det_state = 3  # 异常时回到常态，避免误触发

            # —— 红灯触发：暂停巡线，进入 redlight 对准流程 ——
            if det_state == 0 and line_following_enabled:
                try:
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                except Exception:
                    pass
                line_following_enabled = False
                handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal)
                line_following_enabled = True
                # 再次低头，确保视角回地面
                try:
                    ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
                    time.sleep(0.2)
                except Exception as e:
                    rate.log("warn_pitch", f"[WARN] 绿灯后再次低头失败：{e}", 2.0)

            # —— 蓝线检测（在 vis 上叠加绘制） ——
            line_center, mask = detect_blue_line(vis)

            # —— Obstacle Follow：只在巡线启用且检测到蓝线时融合速度 ——
            x_cmd, z_cmd = 0.0, 0.0
            if line_following_enabled and line_center is not None:
                # 先按巡线计算速度
                x_speed, z_speed = line_following_control(line_center, W)
                x_cmd, z_cmd = x_speed, z_speed

                # 使用最近障碍识别结果做二次约束
                active = (time.time() - last_ts) <= HOLD_SEC
                box = last_box if active else None
                if box is not None:
                    hf = h_frac_from_box(box, H)  # 框高占比
                    if hf >= STOP_H_FRAC:
                        x_cmd = 0.0
                    elif hf >= SLOW_H_FRAC:
                        x_cmd = FOLLOW_SPEED

                    # 在 vis 上绘制障碍框（红=停/黄=跟随/绿=正常）
                    x, y, w, h = box
                    if h <= 1.5:  # 归一化 → 像素
                        x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)
                    color = (0, 0, 255) if x_cmd == 0.0 else ((0, 255, 255) if x_cmd == FOLLOW_SPEED else (0, 255, 0))
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(vis, f"{which_detector or 'vision'} hf={hf:.2f}", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    rate.log("obs", f"[obs] hf={hf:.2f}  → x={x_cmd:.2f}", 0.5)

                # 下发（保持你原来 z 取负号的约定）
                ep_chassis.drive_speed(x=x_cmd, y=0, z=-z_cmd, timeout=0.1)
                rate.log("speed", f"底盘前进速度: {x_cmd:.2f} m/s, 转向速度: {z_cmd:.1f}°/s", 0.5)
            else:
                # 不在巡线或无蓝线 → 停车
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                if not line_following_enabled:
                    rate.log("pause", "红灯流程进行中：巡线暂停", 0.8)
                else:
                    rate.log("no_line", "未检测到蓝线，停止运动", 0.8)

            # 统一显示：左侧主画面 vis，右侧为蓝线 mask
            try:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            except Exception:
                mask_bgr = np.zeros_like(vis)
            h = min(vis.shape[0], mask_bgr.shape[0])
            vis_show = cv2.resize(vis, (int(vis.shape[1]*h/vis.shape[0]), h))
            mask_show = cv2.resize(mask_bgr, (int(mask_bgr.shape[1]*h/mask_bgr.shape[0]), h))
            panel = np.hstack([vis_show, mask_show])
            cv2.imshow(WINDOW_NAME, panel)

            # 单点 waitKey（统一在这里）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 善后
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass
        try:
            ep_gimbal.recenter().wait_for_completed(timeout=2)
        except Exception:
            pass
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            ep_robot.close()
        except Exception:
            pass
