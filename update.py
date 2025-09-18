# -*- coding: utf-8 -*-
# FollowLine + RedLight + Marker（去掉 obs 避障订阅版：降负载/统一GUI/更稳指令超时）

import time
import cv2
import numpy as np
import os
from robomaster import robot

# ======= 控制器（保持）=======
from RedLight import RedLightController    # 你的 RedLight.py
from marker import MarkerCaptureManager    # 你的 marker.py

# ---------------- 调参（保持可调） ----------------
LOWER_BLUE = np.array([100, 50, 20], dtype=np.uint8)
UPPER_BLUE = np.array([140, 200, 200], dtype=np.uint8)

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

# =============== 日志节流（保持不变） ===============
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

# -------------- 拍照目录（供 Marker 保存、或你手动保存） --------------
SHOOT_DIR = os.path.join(os.path.dirname(__file__), "shots")
os.makedirs(SHOOT_DIR, exist_ok=True)

def save_photo(img_bgr, tag="SHOT"):
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
# ====== 模式切换去重 + 微等待（避免抖动/抢指令）======
_CURRENT_MODE = None

def switch_mode(ep_robot, mode_name: str, settle_sec: float = 0.12):
    """只在模式变化时切换，之后微等待让底层稳定。"""
    global _CURRENT_MODE
    if _CURRENT_MODE == mode_name:
        return
    try:
        ep_robot.set_robot_mode(mode=mode_name)
        _CURRENT_MODE = mode_name
        time.sleep(settle_sec)
    except Exception as e:
        print(f"[mode] 切换到 {mode_name} 异常：{e}")

# =============== 初始化（无 obs） ===============
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
        switch_mode(ep_robot, 'chassis_lead')

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
    redlight_ctrl = RedLightController(show_debug=True)  # 打开调试窗口
    redlight_ctrl.ep = ep_robot
    redlight_ctrl.ep_gimbal = ep_gimbal
    redlight_ctrl.ep_camera = ep_camera
    redlight_ctrl.mode_state = 3
    if hasattr(redlight_ctrl, "_last_state"):
        redlight_ctrl._last_state = 3
    if hasattr(redlight_ctrl, "detect_state"):
        redlight_ctrl.detect_state = 3

    # ✅ 已移除 ObstacleFollower 的创建与订阅

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
    """计算底盘运动控制量（保持原公式）"""
    error = (image_width // 2) - line_center_x
    turn = KP_LINE * error / (image_width // 2)  # 归一化转向量 -1..1
    z_speed = turn * 50                           # 转向速度（度/秒）
    return BASE_SPEED, z_speed

# === 恢复巡线：严格回到程序开头的巡线初始状态（不重启视频流） ===
def reset_line_state_after_green(ep_robot, ep_gimbal):
    """
    绿灯(state==3)后调用：
    - 底盘模式恢复为 chassis_lead
    - 云台回正 + 低头到 PITCH_ANGLE
    - 重新允许巡线
    """
    switch_mode(ep_robot, 'chassis_lead')
    try:
        ep_gimbal.recenter(pitch_speed=60, yaw_speed=60).wait_for_completed()
        time.sleep(0.1)
        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
        time.sleep(0.1)
    except Exception as e:
        rate.log("rl_warn_restore", f"[WARN] 恢复巡线时云台设置异常：{e}", 2.0)

    # 标志位：重新允许巡线
    global line_following_enabled
    line_following_enabled = True

# =============== 红灯流程（保持） ===============
def handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal):
    """交给 redlight_ctrl 控制；直到其 mode_state 回到 3。"""
    rate.log("rl_enter", "[红灯] 暂停巡线，进入 RedLight 控制流程...", 1.0)

    # 停车（放宽timeout以防短卡顿复位）
    try:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
    except Exception:
        pass
    # 进入红灯流程前：切 free，让云台完全独立（去重+微等待）
    switch_mode(ep_robot, 'free')

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

# =============== 主流程（巡线 + 红绿灯 + Marker；无 obs） ===============
if __name__ == '__main__':
    # OpenCV 单线程，降低与 SDK/回调的资源竞争
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 初始化
    ep_robot, ep_camera, ep_chassis, ep_gimbal = init_robot()

    # 启动相机（主程序统一管理）
    ep_camera.start_video_stream(display=False)
    time.sleep(1.0)

    # GUI：只用一个窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ← 新增：Marker 管理器（只用云台 yaw 跟随 + 近距离拍照 + 拍后回正）
    marker_mgr = MarkerCaptureManager(
        ep_robot, ep_camera, ep_robot.vision, ep_chassis,
        show_debug=True,            # ✅ 打开 Marker 调试窗口
        window_name="Markers",      # ✅ 独立窗口名
        only_near=True,             # 只认“近”的 marker
        near_width_min=0.10,        # 近距阈值（可按赛场调）
        track_width_min=0.06,       # 跟随参与阈值（远处太小不跟）
        center_tol=0.05,            # 居中判定
        width_thresh_norm=0.10,     # 拍照触发阈值
        kp_yaw=0.5, max_yaw_dps=60.0,   # 跟随力度
        recenter_after_capture=True,    # 拍完回正(到 0,0)
        miss_restart=999999,            # 不在类里重启视频流，交由主程序管理
        save_dir=SHOOT_DIR
    )
    marker_mgr.subscribe()

    # —— Marker 阶段日志去抖 —— #
    mk_last_stage = "IDLE"
    marker_mode_is_free = False

    frame_id = 0
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

            # 可视化画布：统一在 vis 上叠加（红/绿 + 蓝线 + Marker）
            vis = frame.copy()

            # —— 红/绿检测（隔帧跑，降负载；仅视觉绘制，不触发云台） —— #
            det_state = 3
            if frame_id % 2 == 0:
                try:
                    det_state = redlight_ctrl._detect_color_state(frame, draw=vis)  # 0=红,1=绿,3=未知/常态
                except Exception:
                    det_state = 3  # 异常时回到常态，避免误触发

            # —— 红灯触发：暂停巡线，进入 redlight 对准流程 —— #
            if det_state == 0 and line_following_enabled and time.time() >= t_ignore_red_until:
                try:
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                except Exception:
                    pass
                line_following_enabled = False

                # 阻塞运行红灯流程
                handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal)
                # 恢复巡线姿态
                reset_line_state_after_green(ep_robot, ep_gimbal)

                # 重新允许巡线
                line_following_enabled = True

                # 红灯退出后短暂忽略红灯判定，避免“刚退出又触发”
                t_ignore_red_until = time.time() + 1.2

                # 本帧到此结束，从下一帧重新进入主循环
                continue

            # —— 蓝线检测（在 vis 上叠加绘制；ROI加速） —— #
            line_center, mask = detect_blue_line(vis)

            # ====== Marker 跟随/拍照（与巡线解耦） ====== #
            marker_target, marker_stage, marker_save = None, "IDLE", None
            if line_following_enabled:
                # draw=True 以便 Marker 调试窗口实时刷新
                _, marker_target, marker_stage, marker_save = marker_mgr.step(frame=frame, draw=True)

                # 阶段变化日志（FOLLOW/CAPTURING/IDLE）
                if marker_stage != mk_last_stage:
                    rate.log("mk_stage", f"[marker] stage: {mk_last_stage} -> {marker_stage}", 0.3)
                    mk_last_stage = marker_stage

                # 与巡线解耦：Marker 跟随/拍照期间暂时切 free，结束后切回
                try:
                    if marker_stage in ("FOLLOW", "CAPTURING") and not marker_mode_is_free:
                        switch_mode(ep_robot, 'free')
                        rate.log("mk_mode", "[marker] switch -> free（云台独立，底盘不跟）", 0.5)
                        marker_mode_is_free = True
                    elif marker_stage == "IDLE" and marker_mode_is_free:
                        # 切回底盘跟随前，先确保云台停转，避免残余速度带来抖动
                        try:
                            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                        except Exception:
                            pass
                        switch_mode(ep_robot, 'chassis_lead')
                        rate.log("mk_mode", "[marker] switch -> chassis_lead（恢复巡线）", 0.5)
                        marker_mode_is_free = False
                        # 恢复巡线视角（低头角度），等待完成
                        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()

                except Exception as e:
                    rate.log("mk_mode_warn", f"[WARN] Marker 模式切换异常：{e}", 2.0)

                # 锁定/目标信息日志（限频）
                if marker_target is not None:
                    rate.log("mk_lock",
                             f"[marker] lock id={int(marker_target.mid)} "
                             f"cx={marker_target.x:.2f} w={marker_target.w:.2f}", 0.5)

                # 将 marker 的目标框也叠加到主面板 vis（可视化）
                if marker_target is not None:
                    mx, my, mw, mh = marker_target.x, marker_target.y, marker_target.w, marker_target.h
                    x1 = int((mx - mw / 2) * W); y1 = int((my - mh / 2) * H)
                    x2 = int((mx + mw / 2) * W); y2 = int((my + mh / 2) * H)
                    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
                    x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
                    color = (0, 200, 255) if marker_stage != "CAPTURING" else (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.drawMarker(vis, (int(0.5 * W), int(0.5 * H)), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                    cv2.putText(vis, f"MK {marker_stage}", (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 如果本帧刚拍完（类里已回正到(0,0)），把云台 pitch 拉回地面角度，保持巡线视角一致
                if marker_save:
                    rate.log("mk_save", f"[marker] saved: {os.path.basename(marker_save)}", 2.0)
                    try:
                        ep_gimbal.move(pitch=PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
                    except Exception:
                        pass

            # ====== 巡线控制（仅在“启用巡线且检测到蓝线”时） ====== #
            if line_following_enabled and (line_center is not None):
                x_speed, z_speed = line_following_control(line_center, W)
                z_sent = -z_speed  # 坐标系差异：取负号
                ep_chassis.drive_speed(x=x_speed, y=0, z=z_sent, timeout=0.15)
                rate.log("speed", f"底盘前进速度: {x_speed:.2f} m/s, 转向速度: {z_speed:.1f}°/s", 1.0)
            else:
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

    except KeyboardInterrupt:
        pass
    finally:
        # 善后
        try:
            marker_mgr.stop()
        except Exception:
            pass
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass
        try:
            ep_gimbal.recenter().wait_for_completed()
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
