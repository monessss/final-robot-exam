# -*- coding: utf-8 -*-
# FollowLine + RedLight + Marker（主程序版：断线最低优先级 + 柔性减速替代急停 + 提升巡线增益）

import time
import cv2
import numpy as np
import os
from robomaster import robot
from collections import deque
import math
# ======= 控制器（保持你现有的工具类）=======
from RedLight import RedLightController      # 工具类内部红灯时会切 free
from marker import MarkerCaptureManager      # 工具类 step 内只驱动云台 yaw

# ---------------- 调参 ----------------
LOWER_BLUE   = np.array([100, 50, 20], dtype=np.uint8)
UPPER_BLUE   = np.array([140, 200, 200], dtype=np.uint8)

KP_LINE      = 1.0          # ↑ 增益稍加大（原 0.8）
Z_GAIN       = 80.0         # ↑ 角速度放大系数（原 50）
BASE_SPEED   = 0.2
PITCH_ANGLE  = -50

# 统一显示窗口名 / 固定面板尺寸
WINDOW_NAME = "RM View"
PANEL_W, PANEL_H = 960, 360

# 巡线开关（红灯/Marker 流程触发时关闭；流程结束后统一复位再打开）
line_following_enabled = True

# >>> 断线寻找（探前→扫描）参数 —— 低优先级
SEARCH_YAW_DPS       = 45.0
SEARCH_SWEEP_DEG     = 120.0
FOUND_STABLE_FRAMES  = 2
SEARCH_SEG_TIME      = SEARCH_SWEEP_DEG / abs(SEARCH_YAW_DPS)
PROBE_FWD_TIME       = 0.1
PROBE_FWD_SPEED      = 0.1

# —— 柔性速度控制（替代急停）——
class MotionSmoother:
    """
    对所有底盘速度做斜坡限幅，避免“急停/急转”带来的冲击与误差。
    用法：
      motion.send(x_target, z_target)     # 目标速度（m/s, deg/s）
      motion.soft_stop()                  # 柔性停（目标 0,0）
      motion.hard_zero()                  # 真·急停（仅在异常/退出时使用）
    """
    def __init__(self, chassis, x_slew=0.15, z_slew=40.0, timeout=0.12):
        self.chassis = chassis
        self.x_cur = 0.0
        self.z_cur = 0.0
        self.x_slew = float(x_slew)  # 每循环最大 x 变化量（m/s）
        self.z_slew = float(z_slew)  # 每循环最大 z 变化量（deg/s）
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

# RedLight 控制器实例（init_robot 注入硬件）
redlight_ctrl = None

# =============== 日志节流（小工具） ===============
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

# -------------- 拍照目录（供 Marker/手动保存） --------------
SHOOT_DIR = os.path.join(os.path.dirname(__file__), "shots")
os.makedirs(SHOOT_DIR, exist_ok=True)

# =============== 诊断辅助：Marker 详细状态打印 ===============
def dump_marker_diag(marker_mgr, stage_hint=""):
    try:
        tracks = getattr(marker_mgr, "_tracks", {})
        tracks_n = len(tracks) if isinstance(tracks, dict) else (len(tracks) if tracks is not None else 0)
        cooldown_left = max(0.0, float(getattr(marker_mgr, "_cooldown_until", 0.0) - time.time()))
        last_cmd  = getattr(marker_mgr, "_last_cmd_info", "")
        last_save = getattr(marker_mgr, "_last_save_path", None)
        cfg = {
            "only_near": getattr(marker_mgr, "only_near", None),
            "near_width_min": getattr(marker_mgr, "near_width_min", None),
            "track_width_min": getattr(marker_mgr, "track_width_min", None),
            "width_thresh_norm": getattr(marker_mgr, "width_thresh_norm", None),
            "center_tol": getattr(marker_mgr, "center_tol", None),
            "kp_yaw": getattr(marker_mgr, "kp_yaw", None),
            "max_yaw_dps": getattr(marker_mgr, "max_yaw_dps", None),
            "min_yaw_dps": getattr(marker_mgr, "min_yaw_dps", None),
            "deadband_dps": getattr(marker_mgr, "deadband_dps", None),
        }
        print(f"[MK-DIAG] stage={stage_hint} tracks={tracks_n} "
              f"cooldown_left={cooldown_left:.2f}s last_cmd='{last_cmd}' last_save={last_save}")
        print(f"[MK-DIAG] cfg: {cfg}")
        if tracks_n == 0:
            print("[MK-DIAG] 注意：当前 tracks 为空。若未向 _tracks 写入识别结果，"
                  "则 _stable_markers() 返回空集，step() 将一直是 IDLE。")
    except Exception as e:
        print(f"[MK-DIAG] dump 异常：{e}")
def estimate_line_theta_deg(mask, sample_stride=2, min_pts=300):
    """
    返回蓝线相对“竖直向上”的倾角 θ（度，右倾为正、左倾为负）；失败返回 None
    - 使用 fitLine 对线像素点做鲁棒直线拟合，vx, vy 是方向向量
    - 图像坐标 y 向下，因此相对“竖直向上”的角度 = atan2(vx, -vy)
    """
    ys, xs = np.where(mask > 0)
    if xs.size < min_pts:
        return None
    pts = np.column_stack((xs[::sample_stride], ys[::sample_stride])).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    theta = math.degrees(math.atan2(vx, -vy))
    # 常见范围：[-45°, 45°]，值越大“向上越往右偏”
    return theta

# =============== 初始化（不订阅/不使用避障） ===============
def init_robot():
    """初始化机器人组件；底盘跟随云台；云台回正+低头。"""
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal  = ep_robot.gimbal
    ep_vision  = ep_robot.vision

    try:
        print("设置机器人模式为底盘跟随云台(chassis_lead)...")
        ep_robot.set_robot_mode(mode='chassis_lead')
        print("回正云台...")
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.3)
        print(f"低头看地面，pitch={PITCH_ANGLE}°...")
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.3)
    except Exception as e:
        print(f"[init] 云台设置异常: {e}")

    # 注入 RedLightController
    global redlight_ctrl
    redlight_ctrl = RedLightController(show_debug=True)
    redlight_ctrl.ep = ep_robot
    redlight_ctrl.ep_gimbal = ep_gimbal
    redlight_ctrl.ep_camera = ep_camera
    redlight_ctrl.mode_state = 3
    if hasattr(redlight_ctrl, "_last_state"):  redlight_ctrl._last_state = 3
    if hasattr(redlight_ctrl, "detect_state"): redlight_ctrl.detect_state = 3

    return ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision

def read_newest(cam, timeout=0.12):
    """优先 newest；遇到任意异常都返回 None（不让异常冒出主循环）"""
    try:
        return cam.read_cv2_image(strategy="newest", timeout=timeout)
    except Exception:
        try:
            # 退化到默认策略再试一次
            return cam.read_cv2_image(timeout=timeout)
        except Exception:
            return None
                   # 兼容旧版

class FrameGrabber:
    """统一相机取帧入口：优先 newest；失败返回上一帧；带流看门狗。"""
    def __init__(self, cam, restart_thresh=40, restart_cooldown=2.0):
        self.cam = cam
        self.last = None
        self.empty_count = 0
        self._last_restart = 0.0
        self._restart_thresh = int(restart_thresh)          # 连续空帧多少次触发重启（40≈4~5秒）
        self._restart_cooldown = float(restart_cooldown)    # 两次重启之间的最小间隔（秒）

    def _maybe_restart_stream(self):
        now = time.monotonic()
        if self.empty_count >= self._restart_thresh and (now - self._last_restart) > self._restart_cooldown:
            try:
                print("[cam] 连续空帧过多，重启视频流…")
                self.cam.stop_video_stream()
            except Exception:
                pass
            time.sleep(0.2)
            try:
                self.cam.start_video_stream(display=False)
                self._last_restart = time.monotonic()
            except Exception:
                pass
            # 重启后清空计数
            self.empty_count = 0

    def grab(self, retries=2, timeout=0.10, allow_last=True):
        """返回最新帧；若失败若干次则回退到上一帧；绝不抛异常。"""
        for _ in range(max(1, retries)):
            f = read_newest(self.cam, timeout=timeout)
            if f is not None:
                self.last = f
                self.empty_count = 0
                return f
            else:
                self.empty_count += 1
        # 读不到 → 看门狗尝试重启；再回退上一帧（如允许）
        self._maybe_restart_stream()
        if allow_last and self.last is not None:
            return self.last
        return None



# =============== 蓝线检测（下半 ROI） ===============
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

# =============== 统一复位：绿灯/Marker 结束后都调它 ===============
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

# =============== 红灯阻塞流程（改为“快速减速至停”，不再急停） ===============
def handle_redlight_flow(ep_robot, motion: MotionSmoother):
    rate.log("rl_enter", "[红灯] 暂停巡线，进入 RedLight 流程...", 1.0)

    # 目标速度设 0，交给斜坡器快速减速
    motion.soft_stop()

    # 阻塞调用 redlight_ctrl.step()，直到其 mode_state 回到 3
    while True:
        try:
            det_state, mode_state, _ = redlight_ctrl.step()
        except Exception as e:
            print(f"[redlight] step 异常：{e}")
            det_state, mode_state = 3, 3

        # 持续保持“目标 0,0”（柔性停）
        motion.soft_stop()
        rate.log("rl_state", f"[红灯] det={det_state} mode_state={mode_state}", 0.5)

        if mode_state == 3:
            rate.log("rl_exit", "[绿灯] RedLight 流程结束，准备统一复位...", 1.0)
            break
def pick_first_sweep_dir():
    """
    返回初始旋转方向：+1 表示左转(z>0)，-1 表示右转(z<0)
    优先用近几帧“线倾角”中位数；不够时用最近的底盘yaw指令中位数；都没有默认右转(-1)。
    需已维护：theta_hist（近帧倾角），yaw_send_hist（近帧实际z指令）
    """
    if len(theta_hist) >= 3:
        m = float(np.median(theta_hist))
        return -1 if m > 0 else +1    # θ>0 说明线“向上偏右”，先右转更容易命中
    if len(yaw_send_hist) >= 3:
        m = float(np.median(yaw_send_hist))
        return +1 if m > 0 else -1    # 用最近转向作为兜底
    return -1

# =============== Marker 阻塞流程（改为“快速减速至停”） ===============
def handle_marker_flow(ep_robot, motion: MotionSmoother, marker_mgr, ep_camera):
    print("[marker] 暂停巡线，进入 Marker 拍照流程…")

    # 切 free，避免云台 yaw 带动底盘；同时用柔性停降速至 0
    try:
        ep_robot.set_robot_mode(mode='free')
        time.sleep(0.08)
    except Exception:
        pass
    motion.soft_stop()

    save_path = None
    idle_frames = 0
    while True:
        frame = grabber.grab(retries=3, timeout=0.12, allow_last=True)
        if frame is None:
            motion.soft_stop()
            rate.log("mk_frame", "[marker] 暂无可用帧", 0.5)
            continue

        try:
            _, target, stage, save_path = marker_mgr.step(frame=frame, draw=True)
        except Exception as e:
            print(f"[marker] step 异常：{e}")
            target, stage, save_path = None, "IDLE", None

        # 保持“目标 0,0”（柔性停），不再每帧急停
        motion.soft_stop()

        if target is not None:
            rate.log("mk_lock", f"[marker] stage={stage} id={getattr(target, 'mid', -1)} "
                                 f"cx={getattr(target,'x',0):.2f} w={getattr(target,'w',0):.2f}", 0.3)
            idle_frames = 0
        else:
            idle_frames += 1
            if idle_frames % 10 == 0:
                dump_marker_diag(marker_mgr, stage_hint=stage)

        if save_path is not None:
            print(f"[marker] 本轮拍照完成：{os.path.basename(save_path)}")
            try:
                # 结束时让云台速度回零
                marker_mgr.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass

            # 等待 Marker 内部“拍后回正”线程结束（不再刷帧、不做暖机）
            try:
                t0 = time.time()
                cap_timeout = getattr(marker_mgr, "recenter_timeout", 3.0) + 0.5
                while getattr(marker_mgr, "_cap_active", False) and time.time() - t0 < cap_timeout:
                    time.sleep(0.05)
                gyaw = getattr(marker_mgr, "gimbal_yaw", None)
                tol  = getattr(marker_mgr, "recenter_tol_deg", 2.0)
                if gyaw is not None:
                    t1 = time.time()
                    while abs(getattr(marker_mgr, "gimbal_yaw", gyaw)) > tol and time.time() - t1 < 1.0:
                        time.sleep(0.05)
            except Exception as e:
                print(f"[marker] 等待内部回正结束异常: {e}")

            break

    try:
        marker_mgr._locked_id = None
        marker_mgr._cooldown_until = time.time() + 1.0
    except Exception:
        pass

# =============== 主流程 ===============
if __name__ == '__main__':
    # OpenCV 单线程
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 初始化
    ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision = init_robot()
    grabber = FrameGrabber(ep_camera)
    motion = MotionSmoother(ep_chassis, x_slew=0.15, z_slew=40.0, timeout=0.12)

    # 启动相机
    try:
        ep_camera.start_video_stream(display=False)
        time.sleep(0.8)
    except Exception as e:
        print(f"[cam] start_video_stream 警告：{e}")

    # GUI：一个窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Marker 管理器
    marker_mgr = MarkerCaptureManager(
        ep_robot=ep_robot, ep_camera=ep_camera, ep_vision=ep_vision, ep_chassis=ep_chassis,
        show_debug=True, window_name="Markers",
        only_near=True, near_width_min=0.10, track_width_min=0.06,
        center_tol=0.05, width_thresh_norm=0.10,
        kp_yaw=0.5, max_yaw_dps=60.0,
        recenter_after_capture=False, miss_restart=120, save_dir=SHOOT_DIR,
        # ↓↓↓ 新增
        min_switch_interval_s=0.6,
        area_switch_hysteresis=1.2,
        capture_deadline_s=1.2,
        force_capture_if_near=True
    )

    # 关键：Marker 的拍照线程也走统一入口（不会抛异常）
    marker_mgr.read_frame = lambda timeout=None: grabber.grab(retries=1, timeout=(timeout or 0.12), allow_last=True)

    try:
        marker_mgr.subscribe()
        print("[marker] subscribe 完成（注意：若不向 _tracks 写数据将无识别）")
    except Exception as e:
        print(f"[marker] subscribe 警告：{e}")

    frame_id = 0
    mk_last_stage = "IDLE"
    t_ignore_red_until = 0.0

    # 断线寻找状态（最低优先级）
    searching  = False
    sweep_dir  = +1
    seg_end_ts = 0.0
    found_cnt  = 0
    probing    = False
    probe_end_ts = 0.0
    scan_pass = 0  # 本轮“左右旋转”已完成的段数（0/1/2）
    theta_hist = deque(maxlen=12)  # 近几帧线倾角历史（度）
    yaw_send_hist = deque(maxlen=12)  # 近几帧实际发送给底盘的 z（deg/s）
    try:
        while True:
            # 取帧
            frame = grabber.grab(retries=3, timeout=0.06, allow_last=True)
            if frame is None:
                # 极端情况下连“上一帧”也没有（刚上电/流断），先保持柔性停并继续尝试
                rate.log("cam_empty", "[camera] 暂无可用帧（newest/last 均为空）", 1.0)
                motion.soft_stop()
                continue

            frame_id += 1
            H, W = frame.shape[:2]
            vis = frame.copy()

            # 红/绿检测（仅用于触发红灯流程）
            det_state = 3
            SEARCH_GUARD_HYST = 0.4
            guard_drop_ts = None
            try:
                det_state = redlight_ctrl._detect_color_state(frame, draw=vis)  # 0=红,1=绿,3=常态
            except Exception as e:
                rate.log("rl_detect_err", f"[redlight] _detect_color_state 异常：{e}", 1.0)
                det_state = 3

            # 红灯触发 → 快速减速至停 + 阻塞流程 + 统一复位
            if det_state == 0 and line_following_enabled and time.time() >= t_ignore_red_until:
                line_following_enabled = False
                searching = probing = False
                found_cnt = 0

                handle_redlight_flow(ep_robot, motion)
                reset_line_state_after_green(ep_robot, ep_gimbal)
                line_following_enabled = True
                t_ignore_red_until = time.time() + 1.2
                continue

            # 蓝线检测
            line_center, mask = detect_blue_line(vis)

            # Marker 触发：进入拍照流程（柔性停）
            marker_target, marker_stage, marker_save = None, "IDLE", None
            if line_following_enabled:
                try:
                    _, marker_target, marker_stage, marker_save = marker_mgr.step(frame=frame, draw=True)
                except Exception as e:
                    rate.log("mk_step_err", f"[marker] step 异常：{e}", 0.5)
                    marker_target, marker_stage, marker_save = None, "IDLE", None

                if marker_stage != mk_last_stage:
                    rate.log("mk_stage", f"[marker] stage: {mk_last_stage} -> {marker_stage}", 0.2)
                    mk_last_stage = marker_stage

                if marker_stage == "IDLE" and (frame_id % 10 == 0):
                    dump_marker_diag(marker_mgr, stage_hint=marker_stage)

                if marker_stage in ("FOLLOW", "CAPTURING"):
                    line_following_enabled = False
                    searching = probing = False
                    found_cnt = 0
                    prev_chassis = marker_mgr.ep_chassis
                    marker_mgr.ep_chassis = None  # 禁用“底盘兜底转向”
                    try:
                        handle_marker_flow(ep_robot, motion, marker_mgr, ep_camera)
                        reset_line_state_after_green(ep_robot, ep_gimbal)
                        time.sleep(1.0)
                        line_following_enabled = True
                        t_ignore_red_until = time.time() + 1.2
                    finally:
                        marker_mgr.ep_chassis = prev_chassis
                    continue

                # 可视化叠加
                if marker_target is not None:
                    mx, my, mw, mh = marker_target.x, marker_target.y, marker_target.w, marker_target.h
                    x1 = int((mx - mw / 2) * W); y1 = int((my - mh / 2) * H)
                    x2 = int((mx + mw / 2) * W); y2 = int((my + mh / 2) * H)
                    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
                    x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
                    color = (0, 200, 255) if marker_stage != "CAPTURING" else (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.drawMarker(vis, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                    cv2.putText(vis, f"MK {marker_stage}", (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD
            try:
                cv2.putText(vis, f"LF={'ON' if line_following_enabled else 'OFF'}  RL={det_state}  MK={mk_last_stage}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                tracks_n = len(getattr(marker_mgr, "_tracks", {}) or [])
                cv2.putText(vis, f"MK_tracks={tracks_n}", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            except Exception:
                pass

            # ========== 巡线 + 断线寻找（最低优先级） ==========
            search_guard = (line_following_enabled and marker_stage == "IDLE" and det_state != 0)

            if not line_following_enabled:
                searching = probing = False
                found_cnt = 0
                motion.soft_stop()

            else:
                if line_center is not None:
                    if searching or probing:
                        found_cnt += 1
                        if found_cnt < FOUND_STABLE_FRAMES:
                            # 继续当前动作：探前或旋转（不改模式）
                            if searching:
                                motion.send(0.0, sweep_dir * SEARCH_YAW_DPS)
                            elif probing:
                                motion.send(PROBE_FWD_SPEED, 0.0)
                        else:
                            # —— 确认找回线，收尾 —— #
                            searching = probing = False
                            found_cnt = 0
                            scan_pass = 0
                            motion.soft_stop()
                            print("[SEARCH] 蓝线已找到，恢复巡线")

                    # 正常巡线
                    x_cmd, z_cmd = line_following_control(line_center, W)
                    motion.send(x_cmd, -z_cmd)
                    theta = estimate_line_theta_deg(mask)
                    if theta is not None:
                        theta_hist.append(theta)
                    yaw_send_hist.append(-z_cmd)
                    rate.log("speed", f"底盘前进速度: {x_cmd:.2f} m/s, 转向速度: {z_cmd:.1f}°/s", 1.0)



                else:

                    # 当前帧没看到线

                    if not search_guard:

                        now = time.monotonic()

                        if searching or probing:

                            # HYST：短暂允许继续当前动作，避免抖动打断

                            guard_drop_ts = guard_drop_ts or now

                            if now - guard_drop_ts < SEARCH_GUARD_HYST:

                                if searching:

                                    motion.send(0.0, sweep_dir * SEARCH_YAW_DPS)

                                elif probing:

                                    motion.send(PROBE_FWD_SPEED, 0.0)

                                continue

                        # 超过 HYST → 取消当前“探前/旋转”，静止

                        guard_drop_ts = None

                        searching = probing = False

                        found_cnt = 0

                        scan_pass = 0

                        motion.soft_stop()

                        rate.log("no_line", "未检测到蓝线，保持静止（柔性停）", 1.2)
                    else:
                        # ============ 允许搜索：前进一小段 → 停车 → 左右旋转（两段） ============
                        HALF_SEG_TIME = 0.5 * SEARCH_SEG_TIME  # 左/右各半程，共覆盖 SEARCH_SWEEP_DEG
                        if not searching and not probing:
                            guard_drop_ts = None
                            probing = True
                            probe_end_ts = time.time() + PROBE_FWD_TIME
                            print(f"[SEARCH] 丢线 → 先直行 {PROBE_FWD_TIME:.2f}s，再左右旋转寻找")

                        if probing:
                            if time.time() < probe_end_ts:
                                motion.send(PROBE_FWD_SPEED, 0.0)
                            else:
                                # 停车，转入“左右旋转”
                                motion.soft_stop()
                                probing = False
                                searching = True
                                scan_pass = 0

                                sweep_dir = pick_first_sweep_dir()  # ☆ 用预测方向替代固定 -1
                                seg_end_ts = time.time() + HALF_SEG_TIME
                                motion.send(0.0, sweep_dir * SEARCH_YAW_DPS)

                        elif searching:
                            if time.time() < seg_end_ts:
                                # 继续保持恒速旋转
                                motion.send(0.0, sweep_dir * SEARCH_YAW_DPS)
                            else:
                                # 完成一段，反向；累计段数
                                scan_pass += 1
                                sweep_dir *= -1
                                if scan_pass >= 2:
                                    # 左右两段都扫完仍没找到 → 本轮结束，下一轮会再次“前进一小段”
                                    searching = False
                                    motion.soft_stop()
                                    time.sleep(0.05)  # 小憩让底盘稳住
                                else:
                                    # 还有另一段：继续
                                    seg_end_ts = time.time() + HALF_SEG_TIME
                                    motion.send(0.0, sweep_dir * SEARCH_YAW_DPS)

            # 面板显示
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
        # 善后：这里允许真·急停，保证安全下电
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
        cv2.destroyAllWindows()
        try:
            ep_robot.close()
        except Exception:
            pass
