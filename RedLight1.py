import time
import os
import threading
import cv2
import numpy as np
from robomaster import robot

class RedLightController:
    PITCH_ANGLE = -50
    GREEN_RECENTER_DELAY = 1.0
    YAW_MAX_SPEED_DEG_S = 120.0
    PITCH_MAX_SPEED_DEG_S = 90.0
    TOL_NORM_X = 0.05
    TOL_NORM_Y = 0.05
    team_id = 20
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
    JPEG_QUALITY = 85
    BURST_SHOTS = 2
    BURST_INTERVAL = 0.06
    _POST_SHOT_DELAY_SCALE = 0.3

    def __init__(self, show_debug=True):
        self.show_debug = show_debug
        self._last_state = 3
        self.mode_state = 3
        self._t_green_post_start = None
        self.team_id = 20
        self.ep = None
        self.ep_gimbal = None
        self.ep_camera = None
        self._current_mode = None
        self.SAVE_DIR = os.path.join(os.path.dirname(__file__), 'shots')
        self._shot_red = False
        self._shot_green = False
        self._shoot_thread = None
        self._shoot_lock = threading.Lock()
        self._pre_cache = None

    def _label_for(self, color: str) -> str:
        color = color.lower()
        if color == 'red':
            return f'Team {self.team_id:02d} detects a red light'
        else:
            return f'Team {self.team_id:02d} detects a green light'

    def _draw_lock_marker(self, img, cx, cy, color=(0, 255, 0), text=''):
        if img is None:
            return
        H, W = img.shape[:2]
        cx = int(np.clip(cx, 0, W - 1))
        cy = int(np.clip(cy, 0, H - 1))
        r = 28
        cv2.circle(img, (cx, cy), r, color, 3, cv2.LINE_AA)
        cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 26, 2)
        if text:
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = int(np.clip(cx - tw // 2, 6, W - tw - 6))
            ty = int(np.clip(cy + r + th + 12, th + 12, H - 6))
            cv2.rectangle(img, (tx - 6, ty - th - base - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def _annotated_copy(self, frame, cx, cy, color_name='red'):
        color = (0, 0, 255) if color_name == 'red' else (0, 255, 0)
        out = frame.copy()
        self._draw_lock_marker(out, cx, cy, color=color, text=self._label_for(color_name))
        return out

    def start(self, conn_type='ap'):
        if self.ep is not None and self.ep_camera is not None:
            return
        self.ep = robot.Robot()
        self.ep.initialize(conn_type=conn_type)
        self.ep_gimbal = self.ep.gimbal
        self.ep_camera = self.ep.camera
        self._set_mode('gimbal_lead')
        try:
            self.ep_gimbal.recenter(pitch_speed=120, yaw_speed=120).wait_for_completed(timeout=2)
            time.sleep(0.3)
            self.ep_gimbal.move(pitch=self.PITCH_ANGLE, yaw=0, pitch_speed=30, yaw_speed=30).wait_for_completed()
            time.sleep(0.3)
        except Exception as e:
            print('[WARN] 云台初始化异常：', e)
        self.ep_camera.start_video_stream(display=False)
        print('>> RedLightController 就绪（0=红；1=绿；3=常态）')

    def stop(self):
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

    def _set_mode(self, mode_name: str):
        if self._current_mode == mode_name:
            return
        try:
            self.ep.set_robot_mode(mode=mode_name)
            self._current_mode = mode_name
            time.sleep(0.08)
        except Exception:
            try:
                if mode_name == 'gimbal_lead':
                    self.ep.set_robot_mode(mode=robot.GIMBAL_LEAD)
                    self._current_mode = mode_name
            except Exception:
                pass

    def _save_photo(self, frame, tag='RED'):
        try:
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            for k in range(self.BURST_SHOTS):
                ts = time.strftime('%Y%m%d_%H%M%S')
                ms = int(time.time() % 1 * 1000)
                path = os.path.join(self.SAVE_DIR, f'{tag}_{ts}_{ms:03d}_{k + 1}.jpg')
                ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.JPEG_QUALITY)])
                print(f"[RedLight] 拍照{('成功' if ok else '失败')}：{path}")
                if k + 1 < self.BURST_SHOTS:
                    time.sleep(self.BURST_INTERVAL)
        except Exception as e:
            print('[RedLight] 保存照片异常：', e)

    def _shoot_async(self, frame, tag='RED'):
        with self._shoot_lock:
            if self._shoot_thread is not None and self._shoot_thread.is_alive():
                return False
            img = frame.copy()
            th = threading.Thread(target=self._save_photo, args=(img, tag), daemon=True)
            self._shoot_thread = th
            th.start()
            return True

    def _shoot_done(self):
        with self._shoot_lock:
            return self._shoot_thread is None or not self._shoot_thread.is_alive()

    def step(self):
        frame = self.ep_camera.read_cv2_image(strategy='newest', timeout=0.3)
        if frame is None:
            return (self._last_state, self.mode_state, None)
        H, W = frame.shape[:2]
        vis = frame.copy() if self.show_debug else None
        pre = self._preprocess_frame(frame)
        s = self._detect_color_state(frame, draw=vis, pre=pre)
        if s == 0:
            self._set_mode('chassis_lead')
            if self.mode_state != 0:
                self.mode_state = 0
                self._t_green_post_start = None
                self._shot_red = False
                self._shot_green = False
            rc = self._get_target_center(frame, want='red', pre=pre)
            if rc is not None:
                cx, cy = rc
                if self.show_debug and vis is not None:
                    self._draw_lock_marker(vis, cx, cy, color=(0, 0, 255), text=self._label_for('red'))
                aligned = self._aim_gimbal_to(cx, cy, W, H)
                if (aligned or self._is_aligned(cx, cy, W, H, scale=2.0)) and (not self._shot_red):
                    shot_img = self._annotated_copy(frame, cx, cy, color_name='red')
                    if self._shoot_async(shot_img, tag='RED'):
                        self._shot_red = True
            else:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        elif s == 1:
            if self.mode_state == 0:
                gc = self._get_target_center(frame, want='green', pre=pre)
                if gc is not None:
                    gx, gy = gc
                    if self.show_debug and vis is not None:
                        self._draw_lock_marker(vis, gx, gy, color=(0, 255, 0), text=self._label_for('green'))
                    g_aligned = self._aim_gimbal_to(gx, gy, W, H)
                    if (g_aligned or self._is_aligned(gx, gy, W, H, scale=1.5)) and (not self._shot_green):
                        shot_img = self._annotated_copy(frame, gx, gy, color_name='green')
                        if self._shoot_async(shot_img, tag='GREEN'):
                            self._shot_green = True
                else:
                    self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                if self._shot_green:
                    if self._shoot_done():
                        if self._t_green_post_start is None:
                            self._t_green_post_start = time.time()
                        elif time.time() - self._t_green_post_start >= self.GREEN_RECENTER_DELAY * self._POST_SHOT_DELAY_SCALE:
                            try:
                                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                            except Exception:
                                pass
                            self.mode_state = 3
                            self._t_green_post_start = None
            else:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        else:
            self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        if self.show_debug and vis is not None:
            cv2.imshow('Contours', vis)
            cv2.waitKey(1)
        return (self._last_state, self.mode_state, vis if self.show_debug else None)

    def _is_aligned(self, cx, cy, W, H, scale=1.5):
        ex = cx / float(W) - 0.5
        ey = cy / float(H) - 0.5
        return abs(ex) < self.TOL_NORM_X * scale and abs(ey) < self.TOL_NORM_Y * scale

    @staticmethod
    def _calculate_roundness(contour):
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return 0.0
        return 4.0 * np.pi * (area / (peri * peri))

    def _find_valid_color_contours(self, mask_all, red_mask, green_mask, frame_shape, want='red'):
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
            x0, y0 = (max(0, x), max(0, y))
            x1, y1 = (min(W, x + w), min(H, y + h))
            if x1 <= x0 or y1 <= y0:
                continue
            roi_red = red_mask[y0:y1, x0:x1]
            roi_green = green_mask[y0:y1, x0:x1]
            red_cnt = cv2.countNonZero(roi_red)
            green_cnt = cv2.countNonZero(roi_green)
            if red_cnt == 0 and green_cnt == 0:
                continue
            if want == 'red':
                if red_cnt < green_cnt:
                    continue
            elif green_cnt < red_cnt:
                continue
            outs.append((cnt, x, y, w, h, area))
        return outs

    def _preprocess_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, (self.R1_H_lo, self.R1_S_lo, self.R1_V_lo), (self.R1_H_hi, self.R1_S_hi, self.R1_V_hi))
        mask_r2 = cv2.inRange(hsv, (self.R2_H_lo, self.R2_S_lo, self.R2_V_lo), (self.R2_H_hi, self.R2_S_hi, self.R2_V_hi))
        red_mask = cv2.bitwise_or(mask_r1, mask_r2)
        green_mask = cv2.inRange(hsv, (self.G_H_lo, self.G_S_lo, self.G_V_lo), (self.G_H_hi, self.G_S_hi, self.G_V_hi))
        mask_all = cv2.bitwise_or(red_mask, green_mask)
        kernel = np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE), np.uint8)
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        red_targets = self._find_valid_color_contours(mask_all, red_mask, green_mask, frame.shape, want='red')
        green_targets = self._find_valid_color_contours(mask_all, red_mask, green_mask, frame.shape, want='green')
        self._pre_cache = dict(red_targets=red_targets, green_targets=green_targets, masks=(mask_all, red_mask, green_mask))
        return self._pre_cache

    def _detect_color_state(self, frame, draw=None, pre=None):
        if pre is None:
            pre = self._preprocess_frame(frame)
        red_targets = pre['red_targets']
        green_targets = pre['green_targets']
        if len(red_targets) > 0:
            state = 0
        elif len(green_targets) > 0:
            state = 1
        else:
            state = self._last_state
        if draw is not None:
            for cnt, x, y, w, h, area in green_targets:
                cv2.drawContours(draw, [cnt], -1, (0, 255, 0), 7)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(draw, 'Points: 8', (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(draw, f'Area: {int(area)}', (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            for cnt, x, y, w, h, area in red_targets:
                cv2.drawContours(draw, [cnt], -1, (255, 0, 255), 7)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(draw, 'Points: 8', (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(draw, f'Area: {int(area)}', (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f'STATE: {state} (0=RED, 1=GREEN)', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self._last_state = state
        return state

    def _get_target_center(self, frame, want='red', pre=None):
        if pre is None:
            pre = self._preprocess_frame(frame)
        targets = pre['red_targets'] if want == 'red' else pre['green_targets']
        if not targets:
            return None
        cnt, x, y, w, h, area = max(targets, key=lambda t: t[5])
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        return (int(cx), int(cy))

    def _aim_gimbal_to(self, cx, cy, W, H):
        ex = cx / float(W) - 0.5
        ey = cy / float(H) - 0.5
        aligned = abs(ex) < self.TOL_NORM_X and abs(ey) < self.TOL_NORM_Y
        if not aligned:
            yaw_speed = np.clip(ex * self.YAW_MAX_SPEED_DEG_S * 2.0, -self.YAW_MAX_SPEED_DEG_S, self.YAW_MAX_SPEED_DEG_S)
            pitch_speed = np.clip(-ey * self.PITCH_MAX_SPEED_DEG_S * 2.0, -self.PITCH_MAX_SPEED_DEG_S, self.PITCH_MAX_SPEED_DEG_S)
            self.ep_gimbal.drive_speed(pitch_speed=float(pitch_speed), yaw_speed=float(yaw_speed))
        else:
            self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        return aligned
if __name__ == '__main__':
    ctrl = RedLightController(show_debug=True)
    ctrl.start(conn_type='ap')
    try:
        while True:
            ds, ms, _ = ctrl.step()
            if cv2.waitKey(1) & 255 == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
