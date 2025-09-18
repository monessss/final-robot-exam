# final_project_with_marker.py
# 基于你提供的 final project.py ，加入 marker (ID 1..5) 识别 -> 停车 -> 对准 -> 拍照 功能
# 运行前请确保 robomaster SDK 可用并能连接机器人

import cv2
import numpy as np
import robomaster
from robomaster import robot
import time
import os

# -------------------- 场地/控制参数（按需微调） --------------------
TEAM_ID = 20  # 把它改成你们队号（比如 3 表示 Team 03）

# 蓝色阈值（可按场地光线调节）
LOWER_BLUE = np.array([100, 50, 20])
UPPER_BLUE = np.array([140, 200, 200])

# 巡线控制参数
KP_LINE = 0.8
BASE_SPEED = 0.3
PITCH_ANGLE = -50

# 丢线扫描参数（保留原有行为）
SEARCH_YAW_DPS = 60.0
SEARCH_SWEEP_DEG = 120.0
FOUND_STABLE_FRAMES = 2
SEARCH_SEG_TIME = SEARCH_SWEEP_DEG / abs(SEARCH_YAW_DPS)
PROBE_FWD_TIME = 0.8
PROBE_FWD_SPEED = 0.2

# Marker（拍照）相关参数
MARKER_WIDTH_THRESH_NORM = 0.20  # 只对宽度 >= 屏宽 1/5 的 marker 进行对准拍照（任务要求）
CENTER_TOL = 0.05                # 归一化中心容许误差
MARKER_CONFIRM_WAIT = 1.5        # 对准后稳住再拍的时间（s）
CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# 云台 / 距离控制参数（用于对准）
MAX_YAW_SPEED = 0.3  # 云台 yaw speed 的限制（send_yaw_command 里会 clamp）

# -------------------- 全局缓存（由回调/主循环共享） --------------------
markers = []           # 回调填充的 MarkerInfo 列表（保存归一化 x,y,w,h,id）
markers_ts = 0.0       # 最近一次 markers 更新的时间戳（秒）
captured_marker_ids = set()  # 已拍摄的 marker id，避免重复拍

# -------------------- MarkerInfo 类 & 工具函数 --------------------
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info
    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def w(self): return self._w
    @property
    def h(self): return self._h
    @property
    def id(self):
        try:
            return int(self._info)
        except Exception:
            return None

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

# -------------------- 识别回调（注册到 ep_vision） --------------------
def on_detect_marker(marker_info):
    """
    marker_info: list of detections, each element like (x,y,w,h,info)
    x,y,w,h are normalized (0..1), info is string id
    我们只保留 id in [1..5] 且宽度 >= MARKER_WIDTH_THRESH_NORM 的标志
    """
    global markers, markers_ts
    try:
        markers.clear()
        for item in marker_info:
            # 兼容不同 SDK 返回结构
            if len(item) >= 5:
                x, y, w, h, info = item[:5]
            else:
                continue
            try:
                mid = int(info)
            except Exception:
                continue
            if 1 <= mid <= 5 and w >= MARKER_WIDTH_THRESH_NORM:
                markers.append(MarkerInfo(x, y, w, h, mid))
        if markers:
            markers_ts = time.time()
    except Exception as e:
        print("[WARN] on_detect_marker exception:", e)

# -------------------- 云台优先 yaw 命令（失败回退到底盘转） --------------------
def send_yaw_command(ep_robot, ep_chassis, yaw_speed):
    """
    尝试用云台转 yaw；若失败则回退用底盘原地转（fallback）
    yaw_speed: -0.3..0.3
    """
    yaw_speed = clamp(yaw_speed, -MAX_YAW_SPEED, MAX_YAW_SPEED)
    try:
        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=-yaw_speed)
        return True
    except Exception:
        try:
            z = -yaw_speed * 120.0
            ep_chassis.drive_speed(x=0, y=0, z=z, timeout=0.1)
        except Exception:
            pass
        return False

# -------------------- 巡线检测函数（保持你的原实现） --------------------
def detect_blue_line(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

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
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 12)
    z_speed = turn * 123
    return BASE_SPEED, z_speed

# -------------------- 初始化机器人 --------------------
def init_robot():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal

    try:
        print("设置机器人模式为底盘跟随云台...")
        ep_robot.set_robot_mode(mode='chassis_lead')

        print("回正云台...")
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.4)

        print(f"将云台 pitch 调整为 {PITCH_ANGLE}°（低头看地面）")
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.4)
    except Exception as e:
        print(f"[WARN] gimbal init failed: {e}")

    return ep_robot, ep_camera, ep_chassis, ep_gimbal

# -------------------- 主流程 --------------------
if __name__ == '__main__':
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()

    # 订阅 marker 检测回调
    try:
        ep_vision = ep_robot.vision
        ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
        print("[INFO] 已订阅 marker 检测回调")
    except Exception as e:
        print("[WARN] 无法订阅 marker 检测：", e)

    # 启动视频流（和你原始脚本保持一致）
    try:
        ep_camera.start_video_stream(display=False)
    except Exception:
        pass
    time.sleep(2)

    # 丢线扫描状态
    searching = False
    sweep_dir = +1
    seg_end_ts = 0.0
    found_cnt = 0

    probing = False
    probe_end_ts = 0.0

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                # 如果没拿到帧，允许键盘退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            H, W = img.shape[:2]
            line_center, mask = detect_blue_line(img)

            # 先可视化（和原脚本一致）
            cv2.imshow("Camera View", img)
            cv2.imshow("Blue Line Mask", mask)

            # ---- 先处理丢线 / 探前 / 扫描逻辑（与原脚本一致） ----
            x_speed = 0.0; z_speed = 0.0  # 默认速度防止未定义情况

            if line_center is not None:
                # 找到蓝线：退出搜索/探前
                if searching or probing:
                    found_cnt += 1
                    if found_cnt < FOUND_STABLE_FRAMES:
                        # 仍需稳帧：保持当前动作（保持扫描角速度或探前直行）
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

                # 正常巡线计算速度
                x_speed, z_speed = line_following_control(line_center, W)

            else:
                # 丢线：先探前再扫描，和原脚本一致
                if not searching and not probing:
                    probing = True
                    probe_end_ts = time.time() + PROBE_FWD_TIME
                    print(f"[SEARCH] 丢线 -> 先直行 {PROBE_FWD_TIME:.2f}s 再扫描")
                if probing:
                    if time.time() < probe_end_ts:
                        ep_chassis.drive_speed(x=PROBE_FWD_SPEED, y=0.0, z=0.0, timeout=0.1)
                    else:
                        probing = False
                        searching = True
                        sweep_dir = +1
                        seg_end_ts = time.time() + SEARCH_SEG_TIME
                        found_cnt = 0
                        print("[SEARCH] 探前结束 -> 开始左右扫")
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)
                elif searching:
                    if time.time() < seg_end_ts:
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)
                    else:
                        sweep_dir *= -1
                        seg_end_ts = time.time() + SEARCH_SEG_TIME
                        ep_chassis.drive_speed(x=0.0, y=0.0, z=sweep_dir * SEARCH_YAW_DPS, timeout=0.1)

            # ---- 在正常巡线或扫描中加入 marker 检测与处理 ----
            # 只有当我们处于“已检测到蓝线并在按循线速度行进” 或者 即将行进的情况下，才触发 marker 拍照流程
            # （但如果你希望即使在丢线扫描时也拍 marker，可移除下面这个条件）
            # 我们在这里允许：当 markers 存在（最近有更新）且该 marker 符合条件时短暂停车并对准拍照
            snapshot_markers = markers[:]  # 快照，避免回调并发修改
            # 检查是否存在未被捕获且宽度达标的 marker
            target_marker = None
            if snapshot_markers:
                # 选宽度最大的 marker（通常代表最近的）
                mk0 = max(snapshot_markers, key=lambda m: m._w)
                try:
                    mk_id = int(mk0._info)
                except Exception:
                    mk_id = None
                if mk_id is not None and mk_id not in captured_marker_ids and mk0._w >= MARKER_WIDTH_THRESH_NORM:
                    target_marker = mk0

            if target_marker is not None:
                # ---- 执行 marker 对准与拍照（短暂中断巡线） ----
                print(f"[MARKER] 发现目标 Marker ID={target_marker._info}, width={target_marker._w:.3f}，开始对准并拍照")
                # 停车
                try:
                    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
                except Exception:
                    pass

                # 对准：使用云台优先 yaw；如无云台权限回退到底盘原地转
                align_start = time.time()
                aligned = False
                while time.time() - align_start < 4.0:
                    # 刷新 marker 状态（如果回调更新了 markers，就取最新的最大宽度的）
                    snap2 = markers[:]  # 最新快照
                    if snap2:
                        mk_now = max(snap2, key=lambda m: m._w)
                    else:
                        mk_now = target_marker
                    center_error = mk_now._x - 0.5  # 归一化误差
                    yaw_speed = clamp(0.6 * center_error, -MAX_YAW_SPEED, MAX_YAW_SPEED) if 'MAX_YAW_SPEED' in globals() else clamp(0.6*center_error, -0.3, 0.3)
                    # 若已经居中
                    if abs(center_error) < CENTER_TOL:
                        aligned = True
                        break
                    # 发送 yaw 命令（云台优先，失败回落到底盘）
                    send_yaw_command(ep_robot, ep_chassis, yaw_speed)
                    # 小等待以给云台/回调时间更新
                    time.sleep(0.08)

                # 稳定后拍照（若对准失败也尝试拍一次）
                if aligned or abs(center_error) < CENTER_TOL:
                    time.sleep(MARKER_CONFIRM_WAIT)
                # 试图从相机获取一帧（若失败则使用之前的 img）
                try:
                    frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.9)
                    if frame is None:
                        frame = img.copy()
                except Exception:
                    frame = img.copy()

                h2, w2 = frame.shape[:2]
                # 计算像素坐标并画框与文本
                x1 = int((mk_now._x - mk_now._w / 2) * w2)
                y1 = int((mk_now._y - mk_now._h / 2) * h2)
                x2 = int((mk_now._x + mk_now._w / 2) * w2)
                y2 = int((mk_now._y + mk_now._h / 2) * h2)
                x1 = max(0, min(x1, w2 - 1)); y1 = max(0, min(y1, h2 - 1))
                x2 = max(0, min(x2, w2 - 1)); y2 = max(0, min(y2, h2 - 1))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Team {TEAM_ID:02d} detects a marker with ID of {mk_now._info}"
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = max(0, min(x1, w2 - tw - 6))
                ty = max(th + 6, y1 - 8)
                cv2.rectangle(frame, (tx - 4, ty - th - base - 4), (tx + tw + 4, ty + base + 4), (0, 0, 0), -1)
                cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 保存文件（文件名包含队号、marker id、时间戳）
                fname = os.path.join(CAPTURE_DIR, f"Team{TEAM_ID:02d}_marker_{mk_now._info}_{int(time.time())}.jpg")
                try:
                    cv2.imwrite(fname, frame)
                    captured_marker_ids.add(int(mk_now._info))
                    print(f"[SAVE] 已保存 Marker 快照：{fname}")
                except Exception as e:
                    print("[ERROR] 保存 Marker 图像失败：", e)

                # 拍完照后，恢复巡线状态（无需手动继续，循环会按正常巡线计算下发速度）
                # 给一个短暂延时，避免立刻重复检测到同一 marker（若需要可调整）
                time.sleep(0.5)
                continue  # 处理完 marker 后，跳到下一帧循环继续巡线

            # ---- 若没有触发 marker 拍照，则正常下发巡线速度 ----
            try:
                ep_chassis.drive_speed(x=x_speed, y=0, z=-z_speed, timeout=0.1)
            except Exception:
                pass

            # 退出按键
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，开始清理 ...")
    finally:
        # 清理/取消订阅/恢复
        try: ep_vision.unsub_detect_info(name="marker")
        except Exception: pass
        try:
            ep_gimbal.recenter().wait_for_completed(timeout=2)
        except Exception:
            pass
        try: ep_camera.stop_video_stream()
        except Exception: pass
        try: ep_chassis.drive_speed(x=0, y=0, z=0)
        except Exception: pass
        try: ep_robot.close()
        except Exception: pass
        cv2.destroyAllWindows()
