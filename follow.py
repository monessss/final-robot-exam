# -*- coding: utf-8 -*-
# FollowLine + RedLight + Marker + Obstacle-Photo (一体化版)

import time
import cv2
import numpy as np
import os
from robomaster import robot

# ======= 控制器（保持你现有的工具类）=======
from RedLight1 import RedLightController
from marker_end1 import MarkerCaptureManager

# ---------------- 调参 ----------------
LOWER_BLUE   = np.array([100, 92, 40], dtype=np.uint8)
UPPER_BLUE   = np.array([140, 200, 200], dtype=np.uint8)

KP_LINE      = 1.0
Z_GAIN       = 80.0
BASE_SPEED   = 0.35
PITCH_ANGLE  = -50

# —— 避障参数 —— #
MIN_DISTANCE = 0.30   # m：小于此距离认为前方有障碍/小车
OBSTACLE_COOLDOWN_S = 2.5  # m：每次拍照后的冷却，避免重复刷图

WINDOW_NAME = "RM View"
PANEL_W, PANEL_H = 960, 360
line_following_enabled = True
t_ignore_marker_until = 0.0

SEARCH_YAW_DPS       = 60.0
SEARCH_SWEEP_DEG     = 120.0
FOUND_STABLE_FRAMES  = 2
SEARCH_SEG_TIME      = SEARCH_SWEEP_DEG / abs(SEARCH_YAW_DPS)
PROBE_FWD_TIME       = 0.0
PROBE_FWD_SPEED      = 0.0

# —— 全局避障状态（距离回调写入）——
obstacle_detected = False
last_distance_m   = None
t_obstacle_cool_until = 0.0
paused_for_obstacle = False  # 仅用于打印/恢复

# —— 柔性速度控制 —— #
class MotionSmoother:
    def __init__(self, chassis, x_slew=0.15, z_slew=40.0, timeout=0.12):
        self.chassis = chassis
        self.x_cur = 0.0
        self.z_cur = 0.0
        self.x_slew = float(x_slew)
        self.z_slew = float(z_slew)
        self.timeout = float(timeout)

    @staticmethod
    def _step(cur, tgt, step):
        if tgt > cur:  return min(cur + step, tgt)
        if tgt < cur:  return max(cur - step, tgt)
        return cur

    def send(self, x_target: float, z_target: float):
        self.x_cur = self._step(self.x_cur, x_target, self.x_slew)
        self.z_cur = self._step(self.z_cur, z_target, self.z_slew)
        try:
            self.chassis.drive_speed(x=self.x_cur, y=0, z=self.z_cur, timeout=self.timeout)
        except Exception:
            pass

    def soft_stop(self):
        self.send(0.0, 0.0)

    def hard_zero(self):
        self.x_cur, self.z_cur = 0.0, 0.0
        try:
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=self.timeout)
        except Exception:
            pass

redlight_ctrl = None

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

SHOOT_DIR = os.path.join(os.path.dirname(__file__), "shots")
os.makedirs(SHOOT_DIR, exist_ok=True)

def dump_marker_diag(marker_mgr, stage_hint=""):
    return

# —— 避障：距离传感器回调 —— #
def on_distance_data(data):
    # data: 毫米列表，取前方第一个
    global obstacle_detected, last_distance_m
    try:
        d_m = float(data[0]) / 1000.0
    except Exception:
        return
    last_distance_m = d_m
    obstacle_detected = (d_m < MIN_DISTANCE)

def init_robot():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal  = ep_robot.gimbal
    ep_vision  = ep_robot.vision
    ep_sensor  = ep_robot.sensor

    try:
        print("设置机器人模式为底盘跟随云台(chassis_lead)...")
        ep_robot.set_robot_mode(mode='chassis_lead')
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.3)
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.3)
    except Exception as e:
        print(f"[init] 云台设置异常: {e}")

    # 订阅距离
    try:
        ep_sensor.sub_distance(freq=10, callback=on_distance_data)
        print("[sensor] sub_distance @10Hz")
    except Exception as e:
        print(f"[sensor][WARN] sub_distance: {e}")

    global redlight_ctrl
    redlight_ctrl = RedLightController(show_debug=True)
    redlight_ctrl.ep = ep_robot
    redlight_ctrl.ep_gimbal = ep_gimbal
    redlight_ctrl.ep_camera = ep_camera
    redlight_ctrl.mode_state = 3
    if hasattr(redlight_ctrl, "_last_state"):  redlight_ctrl._last_state = 3
    if hasattr(redlight_ctrl, "detect_state"): redlight_ctrl.detect_state = 3

    return ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision, ep_sensor

def read_newest(cam, timeout=0.3):
    try:
        return cam.read_cv2_image(strategy="newest", timeout=timeout)
    except Exception:
        try:
            return cam.read_cv2_image(timeout=timeout)
        except Exception:
            return None

class FrameGrabber:
    def __init__(self, cam):
        self.cam = cam
        self.last = None
    def grab(self, retries=6, timeout=0.10, allow_last=True):
        for _ in range(max(1, retries)):
            try:
                f = read_newest(self.cam, timeout=timeout)
            except Exception:
                f = None
            if f is not None:
                self.last = f
                return f
        if allow_last and self.last is not None:
            return self.last
        return None
    def flush(self, n=4, timeout=0.03):
        for _ in range(max(0, n)):
            _ = read_newest(self.cam, timeout=timeout)

def detect_blue_line(img_bgr):
    H, W = img_bgr.shape[:2]
    y0 = int(H * 0.55)
    roi = img_bgr[y0:, :]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c  = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cv2.rectangle(img_bgr, (x, y + y0), (x + w, y + h + y0), (255, 0, 0), 2)
        cv2.circle(img_bgr, (cx, y0 + y + h // 2), 5, (0, 0, 255), -1)
        return cx, mask
    return None, mask

def line_following_control(line_center_x, image_width):
    error = (image_width // 2) - line_center_x
    turn  = KP_LINE * error / (image_width // 2)
    z_dps = turn * Z_GAIN
    return BASE_SPEED, z_dps

def reset_line_state_after_green(ep_robot, ep_gimbal):
    try:
        ep_robot.set_robot_mode(mode='chassis_lead')
        time.sleep(1.0)
    except Exception as e:
        rate.log("mode_warn", f"[WARN] 设置 chassis_lead 异常：{e}", 2.0)
    try:
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.1)
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.1)
    except Exception as e:
        rate.log("rl_restore", f"[WARN] 恢复巡线时云台设置异常：{e}", 2.0)
    global line_following_enabled
    line_following_enabled = True

def handle_redlight_flow(ep_robot, motion: MotionSmoother):
    rate.log("rl_enter", "[红灯] 暂停巡线，进入 RedLight 流程...", 1.0)
    motion.soft_stop()
    while True:
        try:
            det_state, mode_state, _ = redlight_ctrl.step()
        except Exception as e:
            print(f"[redlight] step 异常：{e}")
            det_state, mode_state = 3, 3
        motion.soft_stop()
        if mode_state == 3:
            rate.log("rl_exit", "[绿灯] RedLight 流程结束，准备统一复位...", 1.0)
            break

# —— 安全调用（你若未在类中加入 cooldown_for，这里兜底）——
def safe_marker_cooldown(marker_mgr, sec: float):
    try:
        marker_mgr.cooldown_for(sec)
    except Exception:
        try:
            marker_mgr._locked_id = None
            marker_mgr._last_lock_ts = 0.0
            marker_mgr._cooldown_until = time.time() + float(sec)
            if hasattr(marker_mgr, "_ao_freeze_until"):
                marker_mgr._ao_freeze_until = time.time() + 0.35
        except Exception:
            pass

# —— 近距离障碍拍照 —— #
def capture_obstacle_snapshot(grabber, distance_m: float, save_dir: str):
    frame = grabber.grab(retries=2, timeout=0.06, allow_last=True)
    if frame is None:
        return None
    H, W = frame.shape[:2]
    # 中心区域画一个红框（传感器没有BBox，这里用标准可视化框）
    x1, x2 = int(W*0.25), int(W*0.75)
    y1, y2 = int(H*0.45), int(H*0.90)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)

    label = f"检测到前方小车  d={distance_m:.2f} m"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pad = 6
    bx1, by1 = x1, max(0, y1- th - base - 2*pad)
    bx2, by2 = x1 + tw + 2*pad, by1 + th + base + 2*pad
    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,0,0), -1)
    cv2.putText(frame, label, (x1+pad, by1+th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, ts, (x1+pad, by1+th+base+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    fname = f"obstacle_robot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(save_dir, fname)
    cv2.imwrite(path, frame)
    print(f"[OBST] saved: {path}")
    return path

# ========= 稳定补丁版 Marker 拍照流程 =========
def handle_marker_flow(ep_robot, motion: MotionSmoother, marker_mgr, ep_camera):
    print("[marker] 暂停巡线，进入 Marker 拍照流程…")

    try:
        ep_robot.set_robot_mode(mode='free')
        time.sleep(0.08)
    except Exception:
        pass
    motion.soft_stop()

    save_path = None
    start_time = time.time()
    MAX_TOTAL_TIMEOUT = 12.0  # 整个流程最长 12 秒

    while True:
        # 若此时出现障碍，立刻退出由主循环处理
        if obstacle_detected:
            print("[marker] 遇到障碍，中断本轮 marker 流程")
            break

        if time.time() - start_time > MAX_TOTAL_TIMEOUT:
            print("[marker] !!! 超时强制退出，避免卡死 !!!")
            break

        frame = grabber.grab(retries=6, timeout=0.10, allow_last=True)
        if frame is None:
            motion.soft_stop()
            rate.log("mk_frame", "[marker] 暂无可用帧", 0.5)
            continue

        try:
            _, target, stage, save_path = marker_mgr.step(frame=frame, draw=True)
        except Exception as e:
            print(f"[marker] step 异常：{e}")
            target, stage, save_path = None, "IDLE", None

        motion.soft_stop()

        if save_path is not None:
            print(f"[marker] 本轮拍照完成：{os.path.basename(save_path)}")
            try:
                marker_mgr.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass

            try:
                t0 = time.time()
                while getattr(marker_mgr, "_cap_active", False) and time.time() - t0 < 3.0:
                    motion.soft_stop()
                    time.sleep(0.05)
            except Exception as e:
                print(f"[marker] 等待内部回正异常: {e}")

            try:
                grabber.flush(n=16, timeout=0.02)
            except Exception:
                pass
            break

    # 退出流程时清掉锁并给冷却
    safe_marker_cooldown(marker_mgr, 2.5)

# =============== 主流程 ===============
if __name__ == '__main__':
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision, ep_sensor = init_robot()
    grabber = FrameGrabber(ep_camera)
    motion = MotionSmoother(ep_chassis)

    try:
        ep_camera.start_video_stream(display=False)
        time.sleep(0.8)
    except Exception as e:
        print(f"[cam] start_video_stream 警告：{e}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    marker_mgr = MarkerCaptureManager(
        ep_robot=ep_robot, ep_camera=ep_camera, ep_vision=ep_vision, ep_chassis=ep_chassis,
        show_debug=False, window_name="Markers",
        only_near=True, near_width_min=0.13, track_width_min=0.08,
        center_tol=0.07, width_thresh_norm=0.10,
        kp_yaw=0.35, max_yaw_dps=45.0,
        deadband_dps=3.0,
        recenter_after_capture=False, miss_restart=999999,
        save_dir=SHOOT_DIR
    )
    try:
        marker_mgr.subscribe()
    except Exception as e:
        print(f"[marker] subscribe 警告：{e}")

    frame_id = 0
    mk_last_stage = "IDLE"
    t_ignore_red_until = 0.0

    searching  = False
    sweep_dir  = +1
    seg_end_ts = 0.0
    found_cnt  = 0
    probing    = False
    probe_end_ts = 0.0
    lf_warmup_until = 0.0

    try:
        while True:
            frame = grabber.grab(retries=3, timeout=0.06, allow_last=True)
            if frame is None:
                motion.soft_stop()
                continue

            frame_id += 1
            H, W = frame.shape[:2]
            vis = frame.copy()

            # ======= 1) 近距障碍最高优先级 ======= #
            if obstacle_detected:
                if not paused_for_obstacle:
                    rate.log("obs_enter", "[obs] 检测到近距离障碍，停车并抓拍…", 0.5)
                    paused_for_obstacle = True

                motion.hard_zero()  # 立刻停
                # 抓拍一次（带冷却）
                if time.time() >= t_obstacle_cool_until:
                    _ = capture_obstacle_snapshot(grabber, last_distance_m or -1.0, SHOOT_DIR)
                    t_obstacle_cool_until = time.time() + OBSTACLE_COOLDOWN_S
                # 拍照后也冷却 marker，避免此刻画面里有 marker 又被吸引
                safe_marker_cooldown(marker_mgr, 2.0)
                t_ignore_marker_until = time.time() + 1.5

                # 画面叠加红色提示
                cv2.putText(vis, f"OBSTACLE! d={last_distance_m:.2f}m", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

                # 显示并继续下一帧（保持停止直到障碍消失）
                try:
                    mask_bgr = np.zeros_like(vis)
                except Exception:
                    mask_bgr = np.zeros_like(vis)
                vis_show  = cv2.resize(vis,      (PANEL_W // 2, PANEL_H))
                mask_show = cv2.resize(mask_bgr, (PANEL_W // 2, PANEL_H))
                panel = np.hstack([vis_show, mask_show])
                cv2.imshow(WINDOW_NAME, panel)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                if paused_for_obstacle:
                    rate.log("obs_exit", "[obs] 障碍物消失，恢复巡线", 0.5)
                    reset_line_state_after_green(ep_robot, ep_gimbal)
                    line_following_enabled = True
                    paused_for_obstacle = False

            # ======= 2) 红灯流程 ======= #
            det_state = 3
            try:
                det_state = redlight_ctrl._detect_color_state(frame, draw=vis)
            except Exception as e:
                rate.log("rl_detect_err", f"[redlight] _detect_color_state 异常：{e}", 1.0)
                det_state = 3

            if det_state == 0 and line_following_enabled and time.time() >= t_ignore_red_until:
                line_following_enabled = False
                searching = probing = False
                found_cnt = 0
                handle_redlight_flow(ep_robot, motion)
                reset_line_state_after_green(ep_robot, ep_gimbal)
                line_following_enabled = True
                t_ignore_red_until = time.time() + 1.2
                continue

            # ======= 3) 蓝线检测 + Marker（带忽略窗口） ======= #
            line_center, mask = detect_blue_line(vis)

            marker_target, marker_stage, marker_save = None, "IDLE", None
            if line_following_enabled:
                try:
                    if time.time() >= t_ignore_marker_until:
                        _, marker_target, marker_stage, marker_save = marker_mgr.step(frame=frame, draw=False)
                except Exception as e:
                    rate.log("mk_step_err", f"[marker] step 异常：{e}", 0.5)

                if marker_stage != mk_last_stage:
                    rate.log("mk_stage", f"[marker] stage: {mk_last_stage} -> {marker_stage}", 0.2)
                    mk_last_stage = marker_stage

                if marker_stage in ("FOLLOW", "CAPTURING"):
                    line_following_enabled = False
                    searching = probing = False
                    found_cnt = 0

                    handle_marker_flow(ep_robot, motion, marker_mgr, ep_camera)

                    # —— 拍照流返回：冷却 + 忽略窗口 —— #
                    safe_marker_cooldown(marker_mgr, 3.0)
                    reset_line_state_after_green(ep_robot, ep_gimbal)
                    time.sleep(1.0)
                    line_following_enabled = True
                    t_ignore_red_until = time.time() + 1.2
                    t_ignore_marker_until = time.time() + 2.0
                    continue

                if marker_target is not None:
                    mx, my, mw, mh = marker_target.x, marker_target.y, marker_target.w, marker_target.h
                    x1 = int((mx - mw / 2) * W); y1 = int((my - mh / 2) * H)
                    x2 = int((mx + mw / 2) * W); y2 = int((my + mh / 2) * H)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)

            # ======= 4) 底盘控制（仅在未暂停时） ======= #
            if line_following_enabled:
                if line_center is not None:
                    x_cmd, z_cmd = line_following_control(line_center, W)
                    motion.send(x_cmd, -z_cmd)
                else:
                    motion.soft_stop()

            # ======= 5) 显示 ======= #
            try:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            except Exception:
                mask_bgr = np.zeros_like(vis)
            vis_show  = cv2.resize(vis,      (PANEL_W // 2, PANEL_H))
            mask_show = cv2.resize(mask_bgr, (PANEL_W // 2, PANEL_H))
            panel = np.hstack([vis_show, mask_show])
            cv2.imshow(WINDOW_NAME, panel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("中断，准备退出…")
    finally:
        try:
            motion.hard_zero()
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
        try:
            ep_sensor.unsub_distance()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            ep_robot.close()
        except Exception:
            pass
