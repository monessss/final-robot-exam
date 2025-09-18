# -*- coding: utf-8 -*-
# FollowLine + RedLight + Marker（主程序版：识别→立刻停车→暂停巡线→只动云台→统一复位）
# 说明：本版仅添加“调试输出/可视化/容错”，未修改控制参数与核心流程。

import time
import cv2
import numpy as np
import os
from robomaster import robot

# ======= 控制器（保持你现有的工具类）=======
from RedLight import RedLightController      # 不改动工具类逻辑（内部在红灯时会切 free）  # contentReference
from marker import MarkerCaptureManager      # 不改动工具类逻辑（step 内只驱动云台 yaw）   # contentReference

# ---------------- 调参（可按场地微调） ----------------
LOWER_BLUE = np.array([100, 50, 20], dtype=np.uint8)
UPPER_BLUE = np.array([140, 200, 200], dtype=np.uint8)

KP_LINE = 0.8
BASE_SPEED = 0.2
PITCH_ANGLE = -50

# 统一显示窗口名 / 固定面板尺寸
WINDOW_NAME = "RM View"
PANEL_W, PANEL_H = 960, 360

# 巡线开关（红灯/Marker 流程触发时关闭；流程结束后统一复位再打开）
line_following_enabled = True

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
    """非侵入式读取 marker 管理器内部状态，帮助定位“为何不触发”"""
    try:
        tracks = getattr(marker_mgr, "_tracks", {})
        tracks_n = len(tracks) if isinstance(tracks, dict) else (len(tracks) if tracks is not None else 0)
        cooldown_left = max(0.0, float(getattr(marker_mgr, "_cooldown_until", 0.0) - time.time()))
        last_cmd = getattr(marker_mgr, "_last_cmd_info", "")
        last_save = getattr(marker_mgr, "_last_save_path", None)

        # 配置快照（便于对照是否“条件太苛刻”）
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

        # 关键线索：当前 marker.py 中 _stable_markers() 仅返回 self._tracks.values()，
        # 但文件内没有任何地方填充 _tracks，且 subscribe() 使用 callback=None。
        # 因此若 tracks=0，target 永远 None，stage=IDLE（不会触发）。  # 参考：marker.py
        if tracks_n == 0:
            print("[MK-DIAG] 注意：当前 tracks 为空。若未向 _tracks 写入识别结果，"
                  "则 _stable_markers() 返回空集，step() 将一直是 IDLE。")
    except Exception as e:
        print(f"[MK-DIAG] dump 异常：{e}")

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
        # 仅初始化时设一次 chassis_lead（运行中不反复切）
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

    # 注入 RedLightController（不二次连接/不自启视频）
    global redlight_ctrl
    redlight_ctrl = RedLightController(show_debug=True)
    redlight_ctrl.ep = ep_robot
    redlight_ctrl.ep_gimbal = ep_gimbal
    redlight_ctrl.ep_camera = ep_camera
    redlight_ctrl.mode_state = 3
    if hasattr(redlight_ctrl, "_last_state"): redlight_ctrl._last_state = 3
    if hasattr(redlight_ctrl, "detect_state"): redlight_ctrl.detect_state = 3

    return ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision

# =============== 蓝线检测（下半 ROI） ===============
def detect_blue_line(img_bgr):
    """检测蓝线并返回中心像素 x；同时返回二值 mask 便于显示。"""
    H, W = img_bgr.shape[:2]
    y0 = int(H * 0.55)
    roi = img_bgr[y0:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c  = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2

        # 可视化叠加在原图坐标
        cv2.rectangle(img_bgr, (x, y + y0), (x + w, y + h + y0), (255, 0, 0), 2)
        cv2.circle(img_bgr, (cx, y0 + y + h // 2), 5, (0, 0, 255), -1)
        return cx, mask
    return None, mask

def line_following_control(line_center_x, image_width):
    """计算底盘前进/转向速度（度/秒）"""
    error  = (image_width // 2) - line_center_x
    turn   = KP_LINE * error / (image_width // 2)
    z_dps  = turn * 50.0
    return BASE_SPEED, z_dps

# =============== 统一复位：绿灯/Marker 结束后都调它 ===============
def reset_line_state_after_green(ep_robot, ep_gimbal):
    """
    - 设回 chassis_lead
    - 云台回正 + 低头到 PITCH_ANGLE（都等待完成）
    - 打开巡线开关
    """
    try:
        ep_robot.set_robot_mode(mode='chassis_lead')
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

# =============== 红灯阻塞流程（立刻停车→只动云台） ===============
def handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal):
    rate.log("rl_enter", "[红灯] 暂停巡线，进入 RedLight 流程...", 1.0)

    # 1) 立刻停车
    try:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
    except Exception:
        pass

    # 2) 阻塞调用 redlight_ctrl.step()，直到其 mode_state 回到 3
    while True:
        try:
            det_state, mode_state, _ = redlight_ctrl.step()   # 工具类内部在红灯时会 set mode=free  # contentReference
        except Exception as e:
            print(f"[redlight] step 异常：{e}")
            det_state, mode_state = 3, 3

        # 保证底盘持续静止（覆盖一切残留）
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.25)
        except Exception:
            pass

        # 调试：显示当前检测状态
        rate.log("rl_state", f"[红灯] det={det_state} mode_state={mode_state}", 0.5)

        if mode_state == 3:
            rate.log("rl_exit", "[绿灯] RedLight 流程结束，准备统一复位...", 1.0)
            break

# =============== Marker 阻塞流程（立刻停车→只动云台） ===============
def handle_marker_flow(ep_robot, ep_chassis, ep_gimbal, marker_mgr, ep_camera):
    """
    进入前：主循环已关闭巡线 & 停车。
    过程：仅调用 marker_mgr.step() 用云台 yaw 对齐并抓拍；底盘持续打 0。
    退出：检测到本轮有新 save_path 即退出，然后统一复位。
    """
    print("[marker] 暂停巡线，进入 Marker 拍照流程…")

    # 进入流程前切一次 free，避免 chassis_lead 下云台 yaw 带动底盘旋转
    try:
        ep_robot.set_robot_mode(mode='free')
        time.sleep(0.08)  # 给底层一点 settle
    except Exception:
        pass

    save_path = None
    idle_frames = 0
    while True:
        frame = ep_camera.read_cv2_image(timeout=0.3)
        if frame is None:
            try:
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.25)
            except Exception:
                pass
            rate.log("mk_frame", "[marker] 读取帧为空", 0.5)
            continue

        # 仅让 Marker 工具类工作（外层压 0 保证底盘绝对不动）
        try:
            _, target, stage, save_path = marker_mgr.step(frame=frame, draw=True)
        except Exception as e:
            print(f"[marker] step 异常：{e}")
            target, stage, save_path = None, "IDLE", None

        # 每帧保证底盘静止
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.25)
        except Exception:
            pass

        # 调试：显示当前阶段、目标、内部状态
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
            # 清掉云台速度更干净
            try:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            break

    # 统一复位（回正+低头+切回 chassis_lead）
    reset_line_state_after_green(ep_robot, ep_gimbal)

# =============== 主流程 ===============
if __name__ == '__main__':
    # OpenCV 单线程，降低与 SDK/回调的资源竞争
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 初始化
    ep_robot, ep_camera, ep_chassis, ep_gimbal, ep_vision = init_robot()

    # 启动相机（主程序统一管理）
    try:
        ep_camera.start_video_stream(display=False)
        time.sleep(0.8)
    except Exception as e:
        print(f"[cam] start_video_stream 警告：{e}")

    # GUI：一个窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Marker 管理器（由主程序喂帧，不自启相机）
    marker_mgr = MarkerCaptureManager(
        ep_robot=ep_robot,
        ep_camera=ep_camera,
        ep_vision=ep_vision,
        ep_chassis=ep_chassis,
        show_debug=True,
        window_name="Markers",
        only_near=True,
        near_width_min=0.10,
        track_width_min=0.06,
        center_tol=0.05,
        width_thresh_norm=0.10,
        kp_yaw=0.5, max_yaw_dps=60.0,
        recenter_after_capture=True,
        miss_restart=999999,
        save_dir=SHOOT_DIR
    )
    try:
        marker_mgr.subscribe()
        print("[marker] subscribe 完成（注意：当前 marker.py 使用 callback=None，若不向 _tracks 写数据将无识别）")  # 诊断提示
    except Exception as e:
        print(f"[marker] subscribe 警告：{e}")

    frame_id = 0
    mk_last_stage = "IDLE"
    t_ignore_red_until = 0.0

    try:
        while True:
            # 取帧
            frame = ep_camera.read_cv2_image(timeout=0.3)
            if frame is None:
                rate.log("cam_empty", "[camera] 空帧", 1.0)
                try:
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
                except Exception:
                    pass
                continue

            frame_id += 1
            H, W = frame.shape[:2]
            vis = frame.copy()

            # —— 红/绿检测（仅用于触发红灯流程；工具类内部会在红灯时切 free） ——  # contentReference
            det_state = 3
            try:
                det_state = redlight_ctrl._detect_color_state(frame, draw=vis)  # 0=红,1=绿,3=常态
            except Exception as e:
                rate.log("rl_detect_err", f"[redlight] _detect_color_state 异常：{e}", 1.0)
                det_state = 3

            # —— 红灯触发：立刻停车→暂停巡线→阻塞流程→统一复位 ——（与 FollowLine 逻辑一致）  # contentReference
            if det_state == 0 and line_following_enabled and time.time() >= t_ignore_red_until:
                try:
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                except Exception:
                    pass
                line_following_enabled = False

                handle_redlight_flow(ep_robot, ep_chassis, ep_gimbal)
                reset_line_state_after_green(ep_robot, ep_gimbal)

                # 重新允许巡线，并在短时间内忽略红灯，避免立刻复触发
                line_following_enabled = True
                t_ignore_red_until = time.time() + 1.2
                continue

            # —— 蓝线检测（叠加到 vis） ——
            line_center, mask = detect_blue_line(vis)

            # —— Marker 触发：发现进入 FOLLOW/CAPTURING → 立刻停车→暂停巡线→阻塞流程→统一复位 ——
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

                # 若长时间 IDLE，周期性输出内部诊断
                if marker_stage == "IDLE" and (frame_id % 10 == 0):
                    dump_marker_diag(marker_mgr, stage_hint=marker_stage)

                if marker_stage in ("FOLLOW", "CAPTURING"):
                    try:
                        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                    except Exception:
                        pass
                    line_following_enabled = False

                    handle_marker_flow(ep_robot, ep_chassis, ep_gimbal, marker_mgr, ep_camera)
                    line_following_enabled = True
                    t_ignore_red_until = time.time() + 1.2
                    continue

                # 可视化叠加（不改模式）
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

            # —— HUD 角落调试文本（帮助快速判断“是否长期 IDLE/是否触发红灯”） ——
            try:
                cv2.putText(vis, f"LF={'ON' if line_following_enabled else 'OFF'}  RL={det_state}  MK={mk_last_stage}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                # 也显示一下 tracks 数（若属性存在）
                tracks_n = len(getattr(marker_mgr, "_tracks", {}) or [])
                cv2.putText(vis, f"MK_tracks={tracks_n}", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            except Exception:
                pass

            # —— 巡线控制 ——（仅在允许巡线且检测到蓝线时）
            if line_following_enabled and line_center is not None:
                x_cmd, z_cmd = line_following_control(line_center, W)
                try:
                    ep_chassis.drive_speed(x=x_cmd, y=0, z=-z_cmd, timeout=0.15)  # 取负号适配坐标
                except Exception as e:
                    rate.log("drive_err", f"[drive] drive_speed 异常：{e}", 0.5)
                rate.log("speed", f"底盘前进速度: {x_cmd:.2f} m/s, 转向速度: {z_cmd:.1f}°/s", 1.0)
            else:
                # 不允许巡线/无蓝线 → 保持静止
                try:
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.25)
                except Exception:
                    pass
                if not line_following_enabled:
                    rate.log("pause", "流程进行中：巡线暂停", 1.2)
                else:
                    rate.log("no_line", "未检测到蓝线，停止运动", 1.2)

            # —— 统一面板显示（左：画面/叠加；右：蓝线mask） ——
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
