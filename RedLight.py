# -*- coding: utf-8 -*-
import time
import os
import threading
import cv2
import numpy as np
from robomaster import robot


class RedLightController:
    """
    红/绿圆识别 + 云台控制（可复用类）

    状态：
      - detect_state: 0=红色圆形；1=绿色圆形；否则沿用上一帧（红优先）
      - mode_state:   0=红锁定；3=常态（回正/切回巡线由外层完成）

    行为（与原逻辑一致）：
      - detect_state==0：设模式为 free，仅云台转动对准红心；对准当帧拍一张 RED 照片。
      - detect_state==1 且之前在红锁定：
            继续仅云台转动，对准绿心；对准当帧拍一张 GREEN 照片；
            **必须等拍照完成**，再开始“拍照后延时”；到点后把 mode_state=3，把控制权交回巡线。
    """

    # ========= 你的原参数（保持不变） =========
    PITCH_ANGLE = -50
    GREEN_RECENTER_DELAY = 1.0
    YAW_MAX_SPEED_DEG_S = 120.0
    PITCH_MAX_SPEED_DEG_S = 90.0
    TOL_NORM_X = 0.05
    TOL_NORM_Y = 0.05

    areaMin = 1500
    AREA_MAX = 70000
    GREEN_AREA_MIN = 1100

    R1_H_lo, R1_H_hi = 0, 10
    R1_S_lo, R1_S_hi = 120, 255
    R1_V_lo, R1_V_hi = 70, 255

    R2_H_lo, R2_H_hi = 170, 179
    R2_S_lo, R2_S_hi = 120, 255
    R2_V_lo, R2_V_hi = 70, 255

    G_H_lo, G_H_hi = 35, 85
    G_S_lo, G_S_hi = 80, 255
    G_V_lo, G_V_hi = 55, 255

    KERNEL_SIZE = 5

    JPEG_QUALITY = 85  # 85% 画质，IO更快
    BURST_SHOTS = 2  # 连拍 2 张
    BURST_INTERVAL = 0.06

    # ========= 新增的内部细化参数（不改你的原值） =========
    # 仅用于“拍照完成后的等待时间”缩短比例（不改变 GREEN_RECENTER_DELAY 本身）
    _POST_SHOT_DELAY_SCALE = 0.3  # 例如把 1.0s 等效为 0.6s；需要再快可改这个比例

    def __init__(self, show_debug=True):
        self.show_debug = show_debug

        # 视觉状态
        self._last_state = 3         # 未检出沿用上一帧，默认常态3
        self.mode_state = 3          # 3=常态；0=红锁定
        self._t_green_post_start = None

        # 设备
        self.ep = None
        self.ep_gimbal = None
        self.ep_camera = None

        # 模式去重
        self._current_mode = None    # "gimbal_lead" / "free" / "chassis_lead"（本类仅用到前两种）

        # 拍照：目录改相对路径 ./shots；异步线程
        self.SAVE_DIR = os.path.join(os.path.dirname(__file__), "shots")
        self._shot_red = False
        self._shot_green = False
        self._shoot_thread = None    # 后台保存照片线程
        self._shoot_lock = threading.Lock()

        # 本帧预处理缓存（图像复用）
        self._pre_cache = None       # dict: {"red_targets":..., "green_targets":..., "masks":...}

    # ========== 生命周期 ==========
    def start(self, conn_type="ap"):
        """连接机器人、设模式（去重）、回正+低头、开启视频流。"""
        self.ep = robot.Robot()
        self.ep.initialize(conn_type=conn_type)
        self.ep_gimbal = self.ep.gimbal
        self.ep_camera = self.ep.camera

        # 模式：云台领航（只在需要时切换；去重）
        self._set_mode("gimbal_lead")

        # 回正+低头
        try:
            self.ep_gimbal.recenter(pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=2)
            time.sleep(0.3)
            self.ep_gimbal.move(pitch=self.PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
            time.sleep(0.3)
        except Exception as e:
            print("[WARN] 云台初始化异常：", e)

        # 开视频流
        self.ep_camera.start_video_stream(display=False)
        print(">> RedLightController 就绪（0=红；1=绿；3=常态）")

    def stop(self):
        """停止视频、回正、断开。"""
        try:
            self.ep_gimbal.recenter(pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=2)
        except Exception:
            pass
        try:
            self.ep_camera.stop_video_stream()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            self.ep.close()
        except Exception:
            pass

    # ========== 模式去重 ==========
    def _set_mode(self, mode_name: str):
        """仅在模式变化时切换机器人模式，避免反复 set。"""
        if self._current_mode == mode_name:
            return
        try:
            self.ep.set_robot_mode(mode=mode_name)
            self._current_mode = mode_name
        except Exception:
            # 兼容写法（主要是 gimbal_lead 常量）
            try:
                if mode_name == "gimbal_lead":
                    self.ep.set_robot_mode(mode=robot.GIMBAL_LEAD)
                    self._current_mode = mode_name
            except Exception:
                pass

    # ========== 异步拍照 ==========
    def _save_photo(self, frame, tag="RED"):
        try:
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            for k in range(self.BURST_SHOTS):
                ts = time.strftime("%Y%m%d_%H%M%S")
                ms = int((time.time() % 1) * 1000)
                path = os.path.join(self.SAVE_DIR, f"{tag}_{ts}_{ms:03d}_{k + 1}.jpg")
                ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.JPEG_QUALITY)])
                print(f"[RedLight] 拍照{'成功' if ok else '失败'}：{path}")
                if k + 1 < self.BURST_SHOTS:
                    time.sleep(self.BURST_INTERVAL)
        except Exception as e:
            print("[RedLight] 保存照片异常：", e)

    def _shoot_async(self, frame, tag="RED"):
        """启动保存线程；如上一次尚未结束，则跳过本次启动，避免堆积。"""
        with self._shoot_lock:
            if self._shoot_thread is not None and self._shoot_thread.is_alive():
                return False
            img = frame.copy()
            th = threading.Thread(target=self._save_photo, args=(img, tag), daemon=True)
            self._shoot_thread = th
            th.start()
            return True

    def _shoot_done(self):
        """当前是否没有拍照线程在运行（已完成或未启动）。"""
        with self._shoot_lock:
            return (self._shoot_thread is None) or (not self._shoot_thread.is_alive())

    # ========== 每步处理 ==========
    def step(self):
        """
        读取一帧并执行视觉+控制。
        返回 (detect_state, mode_state, vis)：
            detect_state: 0=红；1=绿；未检出沿用上一帧
            mode_state:   0=红锁定；3=常态
            vis:          若 show_debug=True 返回标注画面，否则 None
        """
        frame = self.ep_camera.read_cv2_image(strategy="newest", timeout=0.3)
        if frame is None:
            return self._last_state, self.mode_state, None

        H, W = frame.shape[:2]
        vis = frame.copy() if self.show_debug else None

        # —— 预处理（图像复用）——
        pre = self._preprocess_frame(frame)
        s = self._detect_color_state(frame, draw=vis, pre=pre)  # 红优先

        if s == 0:
            # 红灯：切 free（去重），仅云台转动
            self._set_mode("free")

            if self.mode_state != 0:
                self.mode_state = 0
                self._t_green_post_start = None
                self._shot_red = False
                self._shot_green = False

            rc = self._get_target_center(frame, want="red", pre=pre)
            if rc is not None:
                cx, cy = rc
                aligned = self._aim_gimbal_to(cx, cy, W, H)
                # 放宽拍照门槛（仅用于拍照，不改变云台停转阈值）
                if (aligned or self._is_aligned(cx, cy, W, H, scale=2.0)) and not self._shot_red:
                    if self._shoot_async(frame, tag="RED"):
                        self._shot_red = True
            else:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        elif s == 1:
            # 绿灯：只有在“红锁定”阶段才处理
            if self.mode_state == 0:
                gc = self._get_target_center(frame, want="green", pre=pre)
                if gc is not None:
                    g_aligned = self._aim_gimbal_to(gc[0], gc[1], W, H)
                    if (g_aligned or self._is_aligned(gc[0], gc[1], W, H, scale=1.5)) and not self._shot_green:
                        if self._shoot_async(frame, tag="GREEN"):
                            self._shot_green = True
                else:
                    self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

                # —— 必须等待拍照完成 —— 然后启动拍后延时（缩短比例，不改原参数）
                if self._shot_green:
                    if self._shoot_done():
                        # 拍照完成才启动“拍后延时”
                        if self._t_green_post_start is None:
                            self._t_green_post_start = time.time()
                        else:
                            if time.time() - self._t_green_post_start >= (self.GREEN_RECENTER_DELAY * self._POST_SHOT_DELAY_SCALE):
                                # 拍照完成 + 拍后延时到点 → 交还控制权（mode_state=3）
                                self.mode_state = 3
                                self._t_green_post_start = None
                    # 若尚未完成保存：持续等待（不进入延时计时）
            else:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        else:
            # 未检出目标，保持当前姿态
            self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        if self.show_debug and vis is not None:
            cv2.imshow("Contours", vis)
            cv2.waitKey(1)

        return self._last_state, self.mode_state, (vis if self.show_debug else None)

    # ========== —— 视觉/控制细节 —— ==========

    # 仅用于拍照判定的“宽容对准”，不影响云台停转门槛
    def _is_aligned(self, cx, cy, W, H, scale=1.5):
        ex = (cx / float(W)) - 0.5
        ey = (cy / float(H)) - 0.5
        return (abs(ex) < self.TOL_NORM_X * scale) and (abs(ey) < self.TOL_NORM_Y * scale)

    @staticmethod
    def _calculate_roundness(contour):
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return 0.0
        return 4.0 * np.pi * (area / (peri * peri))

    def _find_valid_color_contours(self, mask_all, red_mask, green_mask, frame_shape, want="red"):
        """
        统一形状/圆度/主色筛选：
        - areaMin <= area < AREA_MAX
        - len(approx)==8
        - roundness >= 0.8
        - 主色占优（want='red'→red>=green；'green'→green>=red）
        """
        H, W = frame_shape[:2]
        dyn_min = max(self.areaMin, int(0.001 * W * H))
        outs = []
        contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < dyn_min or area >= self.AREA_MAX:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 8:
                continue

            roundness = self._calculate_roundness(cnt)
            if roundness < 0.8:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                continue

            roi_red   = red_mask[y0:y1, x0:x1]
            roi_green = green_mask[y0:y1, x0:x1]
            red_cnt   = cv2.countNonZero(roi_red)
            green_cnt = cv2.countNonZero(roi_green)
            if red_cnt == 0 and green_cnt == 0:
                continue

            if want == "red":
                if red_cnt < green_cnt:
                    continue
            else:
                if green_cnt < red_cnt:
                    continue

            outs.append((cnt, x, y, w, h, area))
        return outs

    # —— 图像预处理（一次计算，供状态判断与取圆心共享）——
    def _preprocess_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_r1 = cv2.inRange(hsv, (self.R1_H_lo, self.R1_S_lo, self.R1_V_lo), (self.R1_H_hi, self.R1_S_hi, self.R1_V_hi))
        mask_r2 = cv2.inRange(hsv, (self.R2_H_lo, self.R2_S_lo, self.R2_V_lo), (self.R2_H_hi, self.R2_S_hi, self.R2_V_hi))
        red_mask = cv2.bitwise_or(mask_r1, mask_r2)
        green_mask = cv2.inRange(hsv, (self.G_H_lo, self.G_S_lo, self.G_V_lo), (self.G_H_hi, self.G_S_hi, self.G_V_hi))
        mask_all = cv2.bitwise_or(red_mask, green_mask)

        kernel = np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE), np.uint8)
        mask_all   = cv2.morphologyEx(mask_all,   cv2.MORPH_OPEN, kernel, iterations=1)
        mask_all   = cv2.morphologyEx(mask_all,   cv2.MORPH_CLOSE, kernel, iterations=1)
        red_mask   = cv2.morphologyEx(red_mask,   cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask   = cv2.morphologyEx(red_mask,   cv2.MORPH_CLOSE, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        red_targets   = self._find_valid_color_contours(mask_all, red_mask, green_mask, frame.shape, want="red")
        green_targets = self._find_valid_color_contours(mask_all, red_mask, green_mask, frame.shape, want="green")

        self._pre_cache = dict(
            red_targets=red_targets,
            green_targets=green_targets,
            masks=(mask_all, red_mask, green_mask)
        )
        return self._pre_cache

    def _detect_color_state(self, frame, draw=None, pre=None):
        """0=红优先；1=绿；未检出沿用上一帧。使用预处理结果以避免重复计算。"""
        if pre is None:
            pre = self._preprocess_frame(frame)

        red_targets = pre["red_targets"]
        green_targets = pre["green_targets"]

        if len(red_targets) > 0:
            state = 0
        elif len(green_targets) > 0:
            state = 1
        else:
            state = self._last_state

        if draw is not None:
            for cnt, x, y, w, h, area in green_targets:
                cv2.drawContours(draw, cnt, -1, (0, 255, 0), 7)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(draw, "Points: 8", (x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
                cv2.putText(draw, f"Area: {int(area)}", (x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
            for cnt, x, y, w, h, area in red_targets:
                cv2.drawContours(draw, cnt, -1, (255, 0, 255), 7)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(draw, "Points: 8", (x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0), 2)
                cv2.putText(draw, f"Area: {int(area)}", (x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(draw, f"STATE: {state} (0=RED, 1=GREEN)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        self._last_state = state
        return state

    def _get_target_center(self, frame, want="red", pre=None):
        """寻找 want 颜色目标圆心（按相同筛选逻辑），找不到返回 None。使用预处理结果避免重复计算。"""
        if pre is None:
            pre = self._preprocess_frame(frame)

        targets = pre["red_targets"] if want == "red" else pre["green_targets"]
        if not targets:
            return None
        cnt, x, y, w, h, area = max(targets, key=lambda t: t[5])
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        return int(cx), int(cy)

    def _aim_gimbal_to(self, cx, cy, W, H):
        """将云台指向像素点 (cx, cy)，对准后自动停住云台。"""
        ex = (cx / float(W)) - 0.5     # 右正
        ey = (cy / float(H)) - 0.5     # 下正
        aligned = (abs(ex) < self.TOL_NORM_X) and (abs(ey) < self.TOL_NORM_Y)
        if not aligned:
            yaw_speed   = np.clip(ex  * self.YAW_MAX_SPEED_DEG_S * 2.0,  -self.YAW_MAX_SPEED_DEG_S,   self.YAW_MAX_SPEED_DEG_S)
            pitch_speed = np.clip(-ey * self.PITCH_MAX_SPEED_DEG_S * 2.0, -self.PITCH_MAX_SPEED_DEG_S, self.PITCH_MAX_SPEED_DEG_S)
            self.ep_gimbal.drive_speed(pitch_speed=float(pitch_speed), yaw_speed=float(yaw_speed))
        else:
            self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        return aligned


# 可选：独立运行示例
if __name__ == "__main__":
    ctrl = RedLightController(show_debug=True)
    ctrl.start(conn_type="ap")
    try:
        while True:
            ds, ms, _ = ctrl.step()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
