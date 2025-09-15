# -*- coding: utf-8 -*-
import cv2
import time
from queue import Empty  # 捕获 _queue.Empty
from robomaster import robot, vision

# ====== 参数 ======
CENTER_TOL = 0.05          # 对准阈值（归一化），居中±0.05 内算对准
TEAM_ID = 20               # 团队编号，显示为 Team 20
FRAME_W, FRAME_H = 1280, 720  # 仅 MarkerInfo 的像素换算用，实时绘制已改为按帧尺寸
WIDTH_THRESH_NORM = 0.15   # 宽度阈值（归一化）。只有宽度≥0.15 的标志才参与跟随/拍照

# ==== 云台调试状态 ====
gimbal_yaw = None
gimbal_pitch = None
gimbal_angle_ts = 0.0   # 最近一次角度回调的时间戳
last_cmd_info = "N/A"   # 最近一次发出的转动命令描述
last_log_time = 0.0     # 周期性打印的时间戳

# =================== MarkerInfo 类 ===================
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * FRAME_W), int((self._y - self._h / 2) * FRAME_H)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * FRAME_W), int((self._y + self._h / 2) * FRAME_H)

    @property
    def center(self):
        return int(self._x * FRAME_W), int(self._y * FRAME_H)

    @property
    def text(self):
        return self._info


# =================== 全局量（回调写入） ===================
markers = []
current_distance = 0.0  # 米

# =================== 回调 ===================
def on_detect_marker(marker_info):
    """仅保留 1..5 且宽度≥0.15 的标志"""
    global markers
    markers.clear()

    for i in range(len(marker_info)):
        x, y, w, h, info = marker_info[i]   # x,y,w,h 均为归一化(0~1)
        try:
            marker_id = int(info)           # info 是字符形式的数字
        except Exception:
            continue

        print(f"Detected marker with ID {marker_id}, Width: {w:.5f}")

        if 1 <= marker_id <= 5 and w >= WIDTH_THRESH_NORM:
            markers.append(MarkerInfo(x, y, w, h, marker_id))

def on_distance_data(distance_info):
    global current_distance
    if distance_info:
        current_distance = distance_info[0] / 1000.0  # mm -> m

def on_gimbal_angle(angle_info):
    """订阅云台角度: 期望 angle_info 像 [yaw, pitch] 或 {'yaw':..., 'pitch':...}"""
    global gimbal_yaw, gimbal_pitch, gimbal_angle_ts
    try:
        if isinstance(angle_info, (list, tuple)) and len(angle_info) >= 2:
            gimbal_yaw = float(angle_info[0])
            gimbal_pitch = float(angle_info[1])
        elif isinstance(angle_info, dict):
            gimbal_yaw = float(angle_info.get("yaw", gimbal_yaw))
            gimbal_pitch = float(angle_info.get("pitch", gimbal_pitch))
        else:
            return
        gimbal_angle_ts = time.time()
    except Exception:
        pass

# =================== 工具函数 ===================
def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def send_yaw_command(ep_robot, ep_chassis, yaw_speed):
    """
    尝试让云台按 yaw_speed 转动；若失败则回退用底盘原地转。
    yaw_speed：-0.3~0.3（已被 clamp）
    返回 True 表示云台在动，False 表示走了底盘回退。
    """
    global last_cmd_info, last_log_time
    try:
        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=-yaw_speed)
        last_cmd_info = f"GIMBAL yaw_speed={-yaw_speed:.3f}"
        t = time.time()
        if t - last_log_time > 0.5:
            print(f"[CMD] {last_cmd_info}")
            last_log_time = t
        return True
    except Exception:
        z = -yaw_speed * 120.0  # 底盘角速度（把小数放大到合理角速度范围）
        ep_chassis.drive_speed(x=0, y=0, z=z, timeout=0.1)
        last_cmd_info = f"CHASSIS z={z:.1f}"
        t = time.time()
        if t - last_log_time > 0.5:
            print(f"[CMD] {last_cmd_info} (fallback)")
            last_log_time = t
        return False

def safe_start_video_stream(ep_camera, resolution='720p', bitrate=4, fps=None):
    # 兼容多版本签名
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
        if not ok:
            return False

    time.sleep(0.5)
    for _ in range(warm_tries):
        try:
            _ = ep_camera.read_cv2_image(strategy="newest", timeout=warm_timeout)
            return True
        except Empty:
            time.sleep(0.1)

    try: ep_camera.stop_video_stream()
    except Exception: pass
    time.sleep(0.3)
    if not safe_start_video_stream(ep_camera, resolution=None, bitrate=None, fps=None):
        return False
    time.sleep(0.5)
    try:
        _ = ep_camera.read_cv2_image(strategy="newest", timeout=warm_timeout)
        return True
    except Empty:
        return False


# =================== 主程序 ===================
if __name__ == '__main__':
    # ========= 初始化 =========
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor

    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    ep_sensor.sub_distance(freq=10, callback=on_distance_data)

    # 订阅云台角度（若SDK不支持会走except，不影响主流程）
    try:
        ep_robot.gimbal.sub_angle(freq=10, callback=on_gimbal_angle)
        print("[INFO] 已订阅云台角度 gimbal.sub_angle @10Hz")
    except Exception as e:
        print(f"[WARN] gimbal.sub_angle 不可用：{e}")

    ok_warm = start_stream_with_warmup(
        ep_camera,
        resolution='720p',  # 不支持会自动降级
        bitrate=4,          # 不支持会自动忽略
        fps=None,           # 如需尝试帧率可填 30；不支持会被忽略
        warm_tries=15,
        warm_timeout=2.0
    )
    if not ok_warm:
        print("[WARN] 首次启动未拿到视频帧；程序将继续运行并在循环中自动重试。请检查 Wi-Fi/防火墙/VPN。")

    # ========= 控制参数 =========
    TARGET_CENTER = 0.5
    Kp_y = 0.6
    MAX_SPEED = 0.3

    READ_TIMEOUT = 0.9
    MISS_RESTART = 10
    miss = 0

    try:
        while True:
            # —— 读帧（带容错） ——
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=READ_TIMEOUT)
                miss = 0
            except Empty:
                miss += 1
                if miss in (3, 6):
                    time.sleep(0.2)
                if miss >= MISS_RESTART:
                    print("[INFO] 连续取帧失败，重启视频流 ...")
                    try: ep_camera.stop_video_stream()
                    except Exception: pass
                    time.sleep(0.3)
                    start_stream_with_warmup(ep_camera, resolution='720p', bitrate=4, fps=None,
                                             warm_tries=8, warm_timeout=1.5)
                    miss = 0
                    time.sleep(0.2)
                # 没有图像也允许按键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # —— 叠加可视化（按当前帧尺寸换算） ——
            h, w = img.shape[:2]
            snapshot = markers[:]  # 防止回调并发修改，先拍个快照

            for mk in snapshot:
                # 归一化 -> 像素
                x1 = int((mk._x - mk._w / 2) * w)
                y1 = int((mk._y - mk._h / 2) * h)
                x2 = int((mk._x + mk._w / 2) * w)
                y2 = int((mk._y + mk._h / 2) * h)
                cx = int(mk._x * w)
                cy = int(mk._y * h)

                # 边界夹紧，避免越界导致整框被裁掉
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # 1) 画框（绿色）
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 2) 十字准星
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                cv2.line(img, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 1)
                cv2.line(img, (cx, cy - 12), (cx, cy + 12), (0, 255, 0), 1)
                # 3) 标签
                label = f"Team {TEAM_ID:02d} detects a marker with ID of {mk.text}"
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = max(0, min(x1, w - tw - 6))
                ty = max(th + 6, y1 - 8)
                cv2.rectangle(img, (tx - 4, ty - th - base - 4), (tx + tw + 4, ty + base + 4), (0, 0, 0), -1)
                cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # —— 叠加 HUD：显示最近命令 & 云台角度 ——
            hud_lines = []
            hud_lines.append(f"cmd: {last_cmd_info}")
            if gimbal_yaw is not None and gimbal_pitch is not None and gimbal_angle_ts > 0:
                age_ms = int((time.time() - gimbal_angle_ts) * 1000)
                hud_lines.append(f"gimbal yaw={gimbal_yaw:.1f}°  pitch={gimbal_pitch:.1f}°  age={age_ms}ms")
            else:
                hud_lines.append("gimbal angle: N/A")
            y_base = h - 10
            for i, line in enumerate(reversed(hud_lines)):
                cv2.putText(img, line, (10, y_base - i*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            # —— 跟随 & 拍照逻辑 ——
            if markers:
                # 选“宽度最大”的目标更稳
                mk0 = max(markers, key=lambda m: m._w)

                # 水平误差（归一化）
                center_error = mk0._x - TARGET_CENTER

                # 转动控制量（-0.3~0.3）
                yaw_speed = clamp(0.6 * center_error, -MAX_SPEED, MAX_SPEED)  # Kp_y=0.6

                # 条件：已中心 + 宽度≥0.15 —— 先满足才进入等待
                if abs(center_error) < CENTER_TOL and mk0._w >= WIDTH_THRESH_NORM:
                    print("目标接近中心且宽度达标(≥0.15)，等待 2 秒对准...")
                    time.sleep(2.0)

                    # 二次确认（防抖）：再次看一次当前 markers
                    snapshot2 = markers[:]
                    mk2 = max(snapshot2, key=lambda m: m._w) if snapshot2 else mk0

                    if abs(mk2._x - TARGET_CENTER) < CENTER_TOL and mk2._w >= WIDTH_THRESH_NORM:
                        frame = ep_camera.read_cv2_image(strategy="newest", timeout=READ_TIMEOUT)
                        h2, w2 = frame.shape[:2]

                        # 用当前帧尺寸把归一化坐标换成像素坐标
                        x1 = int((mk2._x - mk2._w / 2) * w2)
                        y1 = int((mk2._y - mk2._h / 2) * h2)
                        x2 = int((mk2._x + mk2._w / 2) * w2)
                        y2 = int((mk2._y + mk2._h / 2) * h2)
                        x1 = max(0, min(x1, w2 - 1))
                        y1 = max(0, min(y1, h2 - 1))
                        x2 = max(0, min(x2, w2 - 1))
                        y2 = max(0, min(y2, h2 - 1))

                        # 画框 + 文本（叠到要保存的图上）
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Team {TEAM_ID:02d} detects a marker with ID of {mk2.text}"
                        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        tx = max(0, min(x1, w2 - tw - 6))
                        ty = max(th + 6, y1 - 8)
                        cv2.rectangle(frame, (tx - 4, ty - th - base - 4),
                                      (tx + tw + 4, ty + base + 4), (0, 0, 0), -1)
                        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2)

                        cv2.imwrite(f"captured_marker_{mk2.text}.jpg", frame)
                        print(f"拍照完成！Marker ID: {mk2.text}")
                        break  # 如需连续拍多张，删掉这一行
                    else:
                        print("二次确认未满足条件(中心/宽度)，继续跟随...")

                # 未对准：继续转到居中（统一走带日志的封装）
                else:
                    send_yaw_command(ep_robot, ep_chassis, yaw_speed)

            # —— 键盘：q 退出；a/d 手动小步转动（可选） ——
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                send_yaw_command(ep_robot, ep_chassis, yaw_speed=+0.2)
            elif key == ord('d'):
                send_yaw_command(ep_robot, ep_chassis, yaw_speed=-0.2)

            # —— 显示 ——
            cv2.imshow("Markers", img)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，开始清理 ...")
    finally:
        # ========= 清理 =========
        try: ep_vision.unsub_detect_info(name="marker")
        except Exception: pass
        try: ep_sensor.unsub_distance()
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
