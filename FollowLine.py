# -*- coding: utf-8 -*-
# FollowLine + RedLight + Obstacle-Follow（性能优化版：降负载/统一GUI/更稳指令超时）

import time
import cv2
import numpy as np
import os
from robomaster import robot
from obstacle_follow import ObstacleFollower  # ← 新增
obs = None                                      # ← 新增：作为全局实例
# ======= RedLight 控制器（保持不变）=======
from RedLight import RedLightController  # 你的 RedLight.py

# ---------------- 调参（保持不变） ----------------
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

KP_LINE = 0.8
BASE_SPEED = 0.2
PITCH_ANGLE = -50

# 统一显示窗口名 / 固定面板尺寸（降低GUI负载）
WINDOW_NAME = "RM View"
PANEL_W, PANEL_H = 960, 360

# 巡线开关（红灯触发时关闭；绿灯流程结束后再打开）
line_following_enabled = True

# RedLight 控制器实例（init_robot 注入硬件）
redlight_ctrl = None

# =============== 日志节流（保持不变，略放宽间隔） ===============
class RateLimiter:
    def __init__(self):
        self.last = {}
    def log(self, key: str, msg: str, min_interval: float = 1.0):
        now = time.time()
        t = self.last.get(key, 0.0)
        if now - t >= min_interval:
            print(msg)
            self.last[key] = now

rate = RateLimiter()

# -------------- 拍照配置（主程序控制：硬停后拍一次） --------------
SHOOT_DIR = os.path.join(os.path.dirname(__file__), "shots")
os.makedirs(SHOOT_DIR, exist_ok=True)

def save_photo(img_bgr, tag="STOP"):
    """保存当前画面到 ./shots 目录，文件名含时间戳。返回(路径)或(None)。"""
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        path = os.path.join(SHOOT_DIR, f"{tag}_{ts}_{ms:03d}.jpg")
        ok = cv2.imwrite(path, img_bgr)
        if ok:
            print(f"[shot] 已保存：{path}")
            return path
        else:
            print(f"[shot] 保存失败：{path}")
            return None
    except Exception as e:
        print(f"[shot] 保存异常：{e}")
        return None

# ------------------- Obstacle Follow：新增（参数不破坏原巡线） -------------------
CRUISE_SPEED = BASE_SPEED     # 正常沿线速度 = 你的 BASE_SPEED
FOLLOW_SPEED = 0.12           # 检到前车但未到停车阈值 → 跟随速度
SLOW_H_FRAC  = 0.18           # 目标框高占全帧高度 ≥ 此值 → 减速
STOP_H_FRAC  = 0.50           # 目标框高占全帧高度 ≥ 此值 → 停车
HOLD_SEC     = 0.7           # 识别结果“有效期”，抖动时适当加大

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
    print("[Vision] subscribe failed (no 'robot'/'car'/'people')]")
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
    redlight_ctrl = RedLightController(show_debug=True)  # 统一窗口显示，这里关闭内部窗口
    redlight_ctrl.ep = ep_robot
    redlight_ctrl.ep_gimbal = ep_gimbal
    redlight_ctrl.ep_camera = ep_camera
    redlight_ctrl.mode_state = 3
    if hasattr(redlight_ctrl, "_last_state"):
        redlight_ctrl._last_state = 3
    if hasattr(redlight_ctrl, "detect_state"):
        redlight_ctrl.detect_state = 3

    # 创建并订阅 ObstacleFollower（负责“进入ROI即双拍”）
    global obs
    obs = ObstacleFollower(lateral_half_frac=0.20)  # 中心±25%，连拍间隔0.25s
    obs.subscribe(ep_vision)
    return ep_robot, ep_camera, ep_chassis, ep_gimbal

# =============== 视觉：蓝线（同逻辑，仅下半 ROI 以降负载） ===============
def detect_blue_line(img_bgr):
    """检测蓝线并返回中心位置；在同一张图上做可视化标注"""
    H, W = img_bgr.shape[:2]
    y0 = int(H * 0.55)            # 只看下半视场
    roi = img_bgr[y0:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        line_center_x = x + w // 2

        # 在原图坐标系上绘制
        cv2.rectangle(img_bgr, (x, y + y0), (x + w, y + h + y0), (255, 0, 0), 2)
        cv2.circle(img_bgr, (line_center_x, y0 + y + h // 2), 5, (0, 0, 255), -1)
        return line_center_x, mask

    return None, mask

def line_following_control(line_center_x, image_width):
    """计算底盘运动控制量（保留原公式）"""
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 2)  # 归一化转向量
    z_speed = turn * 50                           # 转向速度（度/秒）
    return BASE_SPEED, z_speed

# === 恢复巡线：严格回到程序开头的巡线初始状态（不重启视频流） ===
def reset_line_state_after_green(ep_robot, ep_gimbal):
    """
    绿灯(state==3)后调用：
    - 底盘模式恢复为 chassis_lead
    - 云台回正 + 低头到 PITCH_ANGLE
    - 重新允许巡线
    - 清空一次障碍检测的缓存，避免沿用上一帧盒子导致误判
    """
    # 1) 模式 & 云台姿态复位 —— 与 init_robot 保持一致
    ep_robot.set_robot_mode(mode='chassis_lead')
    try:
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.1)
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.1)
    except Exception as e:
        rate.log("rl_warn_restore", f"[WARN] 恢复巡线时云台设置异常：{e}", 2.0)

    # 2) 标志位：重新允许巡线
    global line_following_enabled
    line_following_enabled = True

    # 3) 清空障碍检测缓存（上一个框可能刚好很近，避免误触 SLOW/STOP）
    global obs
    try:
        if obs is not None:
            obs.last_box = None
            obs.last_ts  = 0.0
    except Exception:
        pass

# =============== 红灯流程（保持不变） ===============
def handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal):
    """交给 redlight_ctrl 控制；直到其 mode_state 回到 3。"""
    rate.log("rl_enter", "[红灯] 暂停巡线，进入 RedLight 控制流程...", 1.0)

    # 停车（放宽timeout以防短卡顿复位）
    try:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
    except Exception:
        pass

    # 循环调用 step()，直到回到常态 3
    while True:
        det_state, mode_state, _ = redlight_ctrl.step()
        # 期间保障底盘静止
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
        except Exception:
            pass

        if mode_state == 3:
            rate.log("rl_exit", "[绿灯] RedLight 流程结束，准备恢复巡线（回正+低头）...", 1.0)
            break



# =============== 主流程（融合 Obstacle Follow + 性能优化） ===============
if __name__ == '__main__':
    # OpenCV 单线程，降低与 SDK/回调的资源竞争
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 初始化
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()

    # 启动相机
    ep_camera.start_video_stream(display=False)
    time.sleep(1.0)

    # GUI：只用一个窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # 只在水平中心 ±25% 内承认目标（可按需调节）
    LATERAL_HALF_FRAC = 0.20

    frame_id = 0
    prev_obs_stage = "NORMAL"
    last_stop_photo_path = None

    # ====== 新增：追线降温相关状态 ======
    last_line_center = None  # 最新有效的蓝线中心
    z_prev = 0.0  # 上一帧下发前的 z（用于低通）
    stop_cool_until = 0.0  # STOP 退出后的冷却截止时刻（s）
    # ====== 新增：红灯退出后的“忽略红灯”冷却 ======
    t_ignore_red_until = 0.0

    try:

        while True:
            # 取最新帧（缩短阻塞超时）
            frame = ep_camera.read_cv2_image(timeout=0.3)
            if frame is None:
                rate.log("cam_empty", "[camera] 空帧", 1.0)
                continue

            frame_id += 1
            H, W = frame.shape[:2]

            # 可视化画布：统一在 vis 上叠加（红/绿 + 蓝线 + 障碍框）
            vis = frame.copy()
            # 触发“进入中心ROI即连拍两张”（不改速度）
            try:
                if obs is not None:
                    _ = obs.apply(0.0, vis, draw=True)  # 这里传 0.0 即可，apply 不会改速度
            except Exception as e:
                rate.log("obs_apply_warn", f"[WARN] obs.apply 异常: {e}", 2.0)

            # —— 红/绿检测（隔帧跑，降负载；仅视觉绘制，不触发云台） ——
            det_state = 3
            if frame_id % 2 == 0:
                try:
                    det_state = redlight_ctrl._detect_color_state(frame, draw=vis)  # 0=红,1=绿
                except Exception:
                    det_state = 3  # 异常时回到常态，避免误触发

            # —— 红灯触发：暂停巡线，进入 redlight 对准流程 ——
            # —— 红灯触发：暂停巡线，进入 redlight 对准流程 ——
            if det_state == 0 and line_following_enabled and time.time() >= t_ignore_red_until:
                try:
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                except Exception:
                    pass
                line_following_enabled = False

                # 阻塞运行红灯流程
                handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal)
                # 恢复巡线姿态 + 清一次障碍缓存（确保视角回地、避免刚恢复就被旧框拉停）
                reset_line_state_after_green(ep_robot, ep_gimbal)

                # 重新允许巡线
                line_following_enabled = True

                # 红灯退出后 1.5s 内忽略红灯判定，避免“刚退出又触发”
                t_ignore_red_until = time.time() + 1.2

                # 重置一些回归用状态（可选）
                prev_obs_stage = "NORMAL"
                last_line_center = None
                z_prev = 0.0

                # 本帧到此结束，从下一帧重新进入主循环
                continue

                # 再次低头，确保视角回地

            # —— 蓝线检测（在 vis 上叠加绘制；ROI加速） ——
            line_center, mask = detect_blue_line(frame)

            # —— 统一评估“障碍阶段” (NORMAL / SLOW / STOP)，只认中心 ROI 内的目标 ——
            obs_stage = "NORMAL"  # NORMAL / SLOW / STOP
            obs_box_px = None  # 像素坐标框，用于绘制
            obs_hf = 0.0
            detector_name = None

            # 只用 obs 的检测状态（若订阅失败，active_det 会是 False）
            active_det = False
            dt_det = float("inf")
            if (obs is not None) and (obs.last_ts > 0) and (obs.last_box is not None):
                dt_det = time.time() - obs.last_ts
                active_det = dt_det <= HOLD_SEC

            if active_det:
                x, y, w, h = obs.last_box
                detector_name = obs.which_detector
                # 框中心的水平归一化坐标 & 框高占比
                if h <= 1.5:  # 归一化
                    cx_frac = x + w / 2.0
                    obs_hf = h
                    x_px, y_px, w_px, h_px = int(x * W), int(y * H), int(w * W), int(h * H)
                else:  # 像素
                    cx_frac = (x + w / 2.0) / float(W)
                    obs_hf = h / float(H)
                    x_px, y_px, w_px, h_px = int(x), int(y), int(w), int(h)
                # 仅中心ROI内有效
                if abs(cx_frac - 0.5) <= LATERAL_HALF_FRAC:
                    if obs_hf >= STOP_H_FRAC:
                        obs_stage = "STOP"
                    elif obs_hf >= SLOW_H_FRAC:
                        obs_stage = "SLOW"
                    obs_box_px = (x_px, y_px, w_px, h_px)
                else:
                    cv2.putText(vis, "OUT OF ROI", (max(5, int(x_px)), max(20, int(y_px) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


            # ====== 新增：检测 STOP→(SLOW/NORMAL) 的边沿，启动追线冷却 ======
            if prev_obs_stage == "STOP" and obs_stage != "STOP":
                stop_cool_until = time.time() + 0.4  # 0.3~0.5s 可调

            # —— 若已达 STOP 条件：全轴静止并跳过本帧后续（最高优先级） ——
            if obs_stage == "STOP":
                # 可视化（略）
                try:
                    if obs_box_px is not None:
                        cv2.rectangle(vis, (obs_box_px[0], obs_box_px[1]),
                                      (obs_box_px[0] + obs_box_px[2], obs_box_px[1] + obs_box_px[3]),
                                      (0, 0, 255), 3)
                        cv2.putText(vis, f"{(detector_name or 'vision')} hf={obs_hf:.2f} [STOP]",
                                    (obs_box_px[0], max(20, obs_box_px[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception:
                    pass

                # 1) 锁停：每帧都下发 0 速度；timeout 稍长更稳
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                rate.log("obs_stop", "[obs] 条件满足：中心ROI内且距阈值达标 → 全轴静止", 1.0)

                # 2) 仅在“非STOP→STOP”的边沿拍照：先沉降 80~120ms，再用新帧抓拍
                if prev_obs_stage != "STOP":
                    time.sleep(0.1)
                    _ = ep_camera.read_cv2_image(timeout=0.3)  # 丢一帧
                    frame2 = ep_camera.read_cv2_image(timeout=0.3)
                    if frame2 is None:
                        frame2 = vis
                    last_stop_photo_path = save_photo(frame2, tag="STOP")

                # 3) 显示 & 下一帧
                if frame_id % 2 == 0:
                    try:
                        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    except Exception:
                        mask_bgr = np.zeros_like(vis)
                    vis_show = cv2.resize(vis, (PANEL_W // 2, PANEL_H))
                    mask_show = cv2.resize(mask_bgr, (PANEL_W // 2, PANEL_H))
                    panel = np.hstack([vis_show, mask_show])
                    cv2.imshow(WINDOW_NAME, panel)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                prev_obs_stage = obs_stage
                continue

            # —— 巡线：仅在巡线启用且检测到蓝线时融合速度 ——
            # —— 巡线：融合速度 ——
            x_cmd, z_cmd = 0.0, 0.0
            now = time.time()

            # 5.1 记忆最近有效的线中心
            if line_center is not None:
                last_line_center = line_center

            # 5.2 冷却期内若暂时无蓝线，用“上一帧中心”维持转向；否则按原逻辑
            _use_center = line_center
            if _use_center is None and last_line_center is not None and now < stop_cool_until:
                _use_center = last_line_center

            if line_following_enabled and _use_center is not None:
                # 先按巡线计算速度
                x_speed, z_speed = line_following_control(_use_center, W)

                # 5.3 冷却期对 z 做“软限幅 + 低通”，避免瞬时大幅追线
                if now < stop_cool_until:
                    # 软限幅 ±25°/s（可调）
                    z_speed = max(-25.0, min(25.0, z_speed))
                    # 一阶低通：0.7*旧 + 0.3*新（可调）
                    z_cmd = 0.7 * z_prev + 0.3 * z_speed
                else:
                    z_cmd = z_speed
                z_prev = z_cmd

                # 若为 SLOW 阶段，仅限速前进（不改偏航）
                x_cmd = x_speed
                if obs_stage == "SLOW":
                    x_cmd = FOLLOW_SPEED

                # 下发（保留 z 取负号）
                z_sent = -z_cmd
                ep_chassis.drive_speed(x=x_cmd, y=0, z=z_sent, timeout=0.15)

                # 可视化（障碍框颜色同原）
                if obs_box_px is not None:
                    color = (0, 255, 0) if obs_stage == "NORMAL" else (0, 255, 255)
                    cv2.rectangle(vis, (obs_box_px[0], obs_box_px[1]),
                                  (obs_box_px[0] + obs_box_px[2], obs_box_px[1] + obs_box_px[3]),
                                  color, 3)
                    cv2.putText(vis, f"{which_detector or 'vision'} hf={obs_hf:.2f} [{obs_stage}]",
                                (obs_box_px[0], max(20, obs_box_px[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                rate.log("speed", f"底盘前进速度: {x_cmd:.2f} m/s, 转向速度: {z_cmd:.1f}°/s", 1.0)

            else:
                # 不在巡线或（冷却外且）无蓝线 → 静止
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                if not line_following_enabled:
                    rate.log("pause", "红灯流程进行中：巡线暂停", 1.2)
                else:
                    rate.log("no_line", "未检测到蓝线，停止运动", 1.2)

            # 统一显示：左侧主画面 vis，右侧为蓝线 mask（隔帧刷新，固定尺寸）
            if frame_id % 2 == 0:
                try:
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                except Exception:
                    mask_bgr = np.zeros_like(vis)
                vis_show  = cv2.resize(vis,      (PANEL_W // 2, PANEL_H))
                mask_show = cv2.resize(mask_bgr, (PANEL_W // 2, PANEL_H))
                panel = np.hstack([vis_show, mask_show])
                cv2.imshow(WINDOW_NAME, panel)

                # 单点 waitKey（统一这里）
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 记录本帧阶段用于下一帧的边沿检测
            prev_obs_stage = obs_stage

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