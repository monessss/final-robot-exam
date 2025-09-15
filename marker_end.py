# -*- coding: utf-8 -*-
import cv2
import time
from queue import Empty
from robomaster import robot, vision
import os

# ========= 可调参数 =========

CENTER_TOL = 0.05              # 居中阈值(归一化)，想放宽就调大一点
TEAM_ID = 20
FRAME_W, FRAME_H = 1280, 720

WIDTH_THRESH_NORM = 0.10       # 触发拍照的宽度阈值(归一化)
TRACK_WIDTH_MIN   = 0.06       # 参与“跟随”的最小宽度(远处先跟随)

# 云台角速度（度/秒）——整体偏慢更稳
MAX_YAW_DPS  = 60.0            # 总体速度上限（想更慢/更快就改）
MIN_YAW_DPS  = 5.0             # 小于这个可能动不起来
DEADBAND_DPS = 1.0
DIR_YAW = +1                   # 方向反了改成 -1

# 跟踪平滑与粘性
SMOOTH_ALPHA   = 0.35          # EMA 系数（0.25~0.45）
TRACK_HOLD_SEC = 0.30          # 丢帧后轨迹保留时长

# 等待稳定再拍 & 回正速度
PRE_CAPTURE_DELAY = 0.5        # 已对准后再等 0.5s 再拍
RECENTER_YAW_SPEED   = 60      # 回正速度（慢一点更稳）
RECENTER_PITCH_SPEED = 60

# ========= 状态量 =========
gimbal_yaw = None
gimbal_pitch = None
gimbal_angle_ts = 0.0
last_cmd_info = "N/A"
last_log_time = 0.0

# 回调 -> 平滑轨迹：id -> {'x','y','w','h','ts'}
tracks = {}

# 已经拍过的 ID（不会再次触发）
captured_ids = set()

# 目标粘性
locked_id = None
last_lock_ts = 0.0

# ========== 数据结构 ==========
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x = x; self._y = y; self._w = w; self._h = h; self._info = info
    @property
    def text(self): return self._info

# ========== 回调 ==========
def on_detect_marker(marker_info):
    """回调里做 EMA 平滑，主循环读取稳定轨迹。"""
    now = time.time()
    for i in range(len(marker_info)):
        x, y, w, h, info = marker_info[i]
        try:
            marker_id = int(info)
        except Exception:
            continue

        # 只要 1..5；太小的先不参与跟随
        if not (1 <= marker_id <= 5):
            continue
        if w < TRACK_WIDTH_MIN:
            continue

        if marker_id not in tracks:
            tracks[marker_id] = {'x': x, 'y': y, 'w': w, 'h': h, 'ts': now}
        else:
            t = tracks[marker_id]; a = SMOOTH_ALPHA
            t['x'] = a*x + (1-a)*t['x']
            t['y'] = a*y + (1-a)*t['y']
            t['w'] = a*w + (1-a)*t['w']
            t['h'] = a*h + (1-a)*t['h']
            t['ts'] = now

        # 调试打印（可注释）
        print(f"Detected marker ID={marker_id} | w={w:.3f} x={x:.3f} y={y:.3f}")

def on_gimbal_angle(angle_info):
    global gimbal_yaw, gimbal_pitch, gimbal_angle_ts
    try:
        if isinstance(angle_info, (list, tuple)) and len(angle_info) >= 2:
            gimbal_yaw = float(angle_info[0]); gimbal_pitch = float(angle_info[1])
        elif isinstance(angle_info, dict):
            gimbal_yaw = float(angle_info.get("yaw", gimbal_yaw))
            gimbal_pitch = float(angle_info.get("pitch", gimbal_pitch))
        else:
            return
        gimbal_angle_ts = time.time()
    except Exception:
        pass

# ========== 工具 ==========
def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def send_yaw_command(ep_robot, ep_chassis, yaw_speed_norm, max_speed_norm=0.25):
    """把归一化速度映射成云台角速度(°/s)，云台异常时回退到底盘。"""
    global last_cmd_info, last_log_time
    unit = clamp(yaw_speed_norm / max(1e-6, max_speed_norm), -1.0, 1.0)
    yaw_dps = unit * MAX_YAW_DPS

    if abs(yaw_dps) < DEADBAND_DPS:
        last_cmd_info = "IDLE (deadband)"
        return True
    if abs(yaw_dps) < MIN_YAW_DPS:
        yaw_dps = MIN_YAW_DPS * (1 if yaw_dps > 0 else -1)

    yaw_cmd = DIR_YAW * yaw_dps
    try:
        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=yaw_cmd)
        last_cmd_info = f"GIMBAL yaw={yaw_cmd:.1f} dps"
    except Exception:
        ep_chassis.drive_speed(x=0, y=0, z=yaw_cmd, timeout=0.1)
        last_cmd_info = f"CHASSIS z={yaw_cmd:.1f}"
    t = time.time()
    if t - last_log_time > 0.5:
        print("[CMD]", last_cmd_info); last_log_time = t
    return True

def recenter_gimbal(ep_robot, wait=True, timeout=3.0, tol_deg=3.0):
    """慢速回正到(0,0)。"""
    try:
        ep_robot.gimbal.recenter(pitch_speed=RECENTER_PITCH_SPEED, yaw_speed=RECENTER_YAW_SPEED)
        print(f"[INFO] gimbal.recenter issued ({RECENTER_YAW_SPEED} dps)")
    except Exception as e:
        print(f"[WARN] gimbal.recenter 不可用：{e}")
        try: ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception: pass
        return

    if not wait:
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        if gimbal_yaw is not None and abs(gimbal_yaw) <= tol_deg:
            print("[INFO] gimbal recentered")
            return
        time.sleep(0.05)
    print("[WARN] recenter 等待超时（可能已接近零位）")

def safe_start_video_stream(ep_camera, resolution='720p', bitrate=4, fps=None):
    try:
        if resolution is not None and bitrate is not None and fps is not None:
            ep_camera.start_video_stream(display=False, resolution=resolution, bitrate=bitrate, fps=fps); return True
    except Exception: pass
    try:
        if resolution is not None and fps is not None:
            ep_camera.start_video_stream(display=False, resolution=resolution, fps=fps); return True
    except Exception: pass
    try:
        if resolution is not None:
            ep_camera.start_video_stream(display=False, resolution=resolution); return True
    except Exception: pass
    try:
        ep_camera.start_video_stream(display=False); return True
    except Exception:
        return False

def start_stream_with_warmup(ep_camera, resolution='720p', bitrate=4, fps=None,
                             warm_tries=15, warm_timeout=2.0):
    ok = safe_start_video_stream(ep_camera, resolution=resolution, bitrate=bitrate, fps=fps)
    if not ok:
        ok = safe_start_video_stream(ep_camera, resolution=None, bitrate=None, fps=None)
        if not ok: return False
    time.sleep(0.5)
    for _ in range(warm_tries):
        try:
            _ = ep_camera.read_cv2_image(strategy="newest", timeout=warm_timeout); return True
        except Empty: time.sleep(0.1)
    try: ep_camera.stop_video_stream()
    except Exception: pass
    time.sleep(0.3)
    if not safe_start_video_stream(ep_camera, resolution=None, bitrate=None, fps=None): return False
    time.sleep(0.5)
    try:
        _ = ep_camera.read_cv2_image(strategy="newest", timeout=warm_timeout); return True
    except Empty:
        return False

# ========== 主程序 ==========
if __name__ == '__main__':
    ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
    ep_vision = ep_robot.vision; ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis

    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    try:
        ep_robot.gimbal.sub_angle(freq=10, callback=on_gimbal_angle)
        print("[INFO] gimbal.sub_angle @10Hz")
    except Exception as e:
        print("[WARN] sub_angle:", e)

    ok_warm = start_stream_with_warmup(ep_camera, resolution='720p', bitrate=4, fps=None,
                                       warm_tries=15, warm_timeout=2.0)
    if not ok_warm:
        print("[WARN] 暖机失败，也会继续运行…")

    TARGET_CENTER = 0.5
    Kp_y = 0.5                 # 稍微温柔一点
    MAX_SPEED_NORM = 0.25      # 归一化速度映射到 MAX_YAW_DPS

    READ_TIMEOUT = 0.9
    MISS_RESTART = 10
    miss = 0

    # 拍完一个后，给一点冷却时间，避免连触发
    cooldown_until = 0.0

    try:
        while True:
            # 取帧
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=READ_TIMEOUT); miss = 0
            except Empty:
                miss += 1
                if miss in (3, 6): time.sleep(0.2)
                if miss >= MISS_RESTART:
                    try: ep_camera.stop_video_stream()
                    except Exception: pass
                    time.sleep(0.3)
                    start_stream_with_warmup(ep_camera, resolution='720p', bitrate=4, fps=None,
                                             warm_tries=8, warm_timeout=1.5)
                    miss = 0; time.sleep(0.2)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            h, w = img.shape[:2]
            now = time.time()

            # 从 tracks 生成稳定目标，剔除超时轨迹
            stable = []
            for mid, t in list(tracks.items()):
                if now - t['ts'] <= TRACK_HOLD_SEC:
                    stable.append(MarkerInfo(t['x'], t['y'], t['w'], t['h'], mid))

            # 可视化：框 + 点 + 文本
            for mk in stable:
                x1 = int((mk._x - mk._w/2) * w); y1 = int((mk._y - mk._h/2) * h)
                x2 = int((mk._x + mk._w/2) * w); y2 = int((mk._y + mk._h/2) * h)
                x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cx, cy = int(mk._x*w), int(mk._y*h)
                cv2.circle(img, (cx,cy), 4, (0,255,0), -1)
                label = f"Team {TEAM_ID:02d} detects a marker with ID of {mk.text}"
                (tw,th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = max(0, min(x1, w - tw - 6)); ty = max(th+6, y1-8)
                cv2.rectangle(img, (tx-4, ty-th-base-4), (tx+tw+4, ty+base+4), (0,0,0), -1)
                cv2.putText(img, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # 过滤：忽略已拍过的 ID
            stable_filtered = [m for m in stable if int(m.text) not in captured_ids]

            # 选目标：优先锁定的 ID；否则选“离中心(0.5)最近”的（最近原则）
            target = None
            cand = None
            if locked_id is not None:
                for m in stable_filtered:
                    if int(m.text) == int(locked_id):
                        cand = m; break
            if cand is None and stable_filtered:
                cand = min(stable_filtered, key=lambda m: abs(m._x - TARGET_CENTER))
                locked_id = cand.text
            if cand is not None:
                last_lock_ts = now
                target = cand

            # 控制与拍照
            if target and now >= cooldown_until:
                center_error = target._x - TARGET_CENTER
                yaw_speed_norm = clamp(Kp_y * center_error, -MAX_SPEED_NORM, MAX_SPEED_NORM)

                # 拍照条件：居中 + 宽度达标
                if abs(center_error) < CENTER_TOL and target._w >= WIDTH_THRESH_NORM:
                    print(f"中心且宽度达标，等待 {PRE_CAPTURE_DELAY:.1f}s 再拍…")
                    time.sleep(PRE_CAPTURE_DELAY)

                    # 再取一帧并保存
                    frame = ep_camera.read_cv2_image(strategy="newest", timeout=READ_TIMEOUT)
                    h2, w2 = frame.shape[:2]
                    x1 = int((target._x - target._w/2) * w2); y1 = int((target._y - target._h/2) * h2)
                    x2 = int((target._x + target._w/2) * w2); y2 = int((target._y + target._h/2) * h2)
                    x1 = max(0, min(x1, w2-1)); y1 = max(0, min(y1, h2-1))
                    x2 = max(0, min(x2, w2-1)); y2 = max(0, min(y2, h2-1))
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    label = f"Team {TEAM_ID:02d} detects a marker with ID of {target.text}"
                    (tw,th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    tx = max(0, min(x1, w2 - tw - 6)); ty = max(th+6, y1-8)
                    cv2.rectangle(frame, (tx-4, ty-th-base-4), (tx+tw+4, ty+base+4), (0,0,0), -1)
                    cv2.putText(frame, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.imwrite(f"captured_marker_{target.text}.jpg", frame)
                    print(f"拍照完成！Marker ID: {target.text}")

                    # 标记为“已完成”，之后不再识别这个 ID
                    captured_ids.add(int(target.text))
                    print(f"[INFO] 已加入忽略列表：{sorted(captured_ids)}")

                    # 停住并慢速回正，然后继续寻找其他 ID
                    try: ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    except Exception: pass
                    recenter_gimbal(ep_robot, wait=True, timeout=3.0, tol_deg=2.0)

                    # 冷却一下再开始下一轮，避免立即再次触发
                    cooldown_until = time.time() + 0.6
                    locked_id = None  # 回正后重新选择最近的
                else:
                    # 继续慢速调整到中心
                    send_yaw_command(ep_robot, ep_chassis, yaw_speed_norm, max_speed_norm=MAX_SPEED_NORM)
            else:
                # 若刚丢失，短时间内保持锁；否则清空锁
                if locked_id and time.time() - last_lock_ts <= TRACK_HOLD_SEC:
                    pass
                else:
                    locked_id = None

            # 手动键位
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('a'): send_yaw_command(ep_robot, ep_chassis, +0.15, MAX_SPEED_NORM)
            elif key == ord('d'): send_yaw_command(ep_robot, ep_chassis, -0.15, MAX_SPEED_NORM)
            elif key == ord('c'):
                captured_ids.clear(); print("[INFO] 清空忽略ID")
            elif key == ord('r'):
                recenter_gimbal(ep_robot, wait=True)

            cv2.imshow("Markers", img)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，清理资源…")
    finally:
        try: ep_vision.unsub_detect_info(name="marker")
        except Exception: pass
        try: ep_robot.gimbal.unsub_angle()
        except Exception: pass
        try: ep_camera.stop_video_stream()
        except Exception: pass
        try: ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
        except Exception: pass
        try: ep_robot.close()
        except Exception: pass
        cv2.destroyAllWindows()
