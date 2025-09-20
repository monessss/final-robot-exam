import os
import cv2
import time
import threading
from queue import Empty
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class MarkerInfo:
    x: float
    y: float
    w: float
    h: float
    mid: int

    @property
    def box_px(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

class MarkerCaptureManager:
    def __init__(self,
                 ep_robot,
                 ep_camera,
                 ep_vision,
                 ep_chassis=None,
                 *,
                 center_tol: float = 0.05,
                 width_thresh_norm: float = 0.10,
                 track_width_min: float = 0.06,
                 near_width_min: float = 0.10,
                 only_near: bool = True,
                 kp_yaw: float = 0.35,
                 max_yaw_dps: float = 45.0,
                 min_yaw_dps: float = 5.0,
                 deadband_dps: float = 3.0,
                 dir_yaw: float = +1.0,
                 smooth_alpha: float = 0.45,
                 track_hold_sec: float = 1.20,
                 lock_min_dwell: float = 1.00,
                 lost_grace_sec: float = 0.80,

                 pre_capture_delay: float = 0.39,
                 recenter_after_capture: bool = True,
                 recenter_yaw_speed: float = 60.0,
                 recenter_pitch_speed: float = 60.0,
                 recenter_tol_deg: float = 2.0,
                 recenter_timeout: float = 3.0,
                 cooldown_s: float = 0.60,

                 resolution: str = '720p',
                 bitrate: Optional[int] = 4,
                 fps: Optional[int] = None,
                 read_timeout: float = 0.9,
                 miss_restart: int = 10,

                 team_id: int = 20,
                 show_debug: bool = False,
                 save_dir: str = 'shots',
                 window_name: str = 'Markers',
                 valid_id_range: Tuple[int, int] = (1, 5),

                 anti_osc_flip_window: float = 0.6,
                 anti_osc_max_flips: int = 3,
                 anti_osc_freeze: float = 0.5,
                 settle_frames: int = 4
                 ):
        self.ep_robot = ep_robot
        self.ep_camera = ep_camera
        self.ep_vision = ep_vision
        self.ep_chassis = ep_chassis


        self.center_tol = center_tol
        self.width_thresh_norm = width_thresh_norm
        self.track_width_min = track_width_min
        self.near_width_min = near_width_min
        self.only_near = only_near
        self.kp_yaw = kp_yaw
        self.max_yaw_dps = max_yaw_dps
        self.min_yaw_dps = min_yaw_dps
        self.deadband_dps = deadband_dps
        self.dir_yaw = dir_yaw
        self.smooth_alpha = smooth_alpha
        self.track_hold_sec = track_hold_sec
        self.pre_capture_delay = pre_capture_delay
        self.recenter_after_capture = recenter_after_capture
        self.recenter_yaw_speed = recenter_yaw_speed
        self.recenter_pitch_speed = recenter_pitch_speed
        self.recenter_tol_deg = recenter_tol_deg
        self.recenter_timeout = recenter_timeout
        self.cooldown_s = cooldown_s
        self.resolution = resolution
        self.bitrate = bitrate
        self.fps = fps
        self.read_timeout = read_timeout
        self.miss_restart = miss_restart
        self.team_id = team_id
        self.show_debug = show_debug
        self.window_name = window_name
        self.valid_id_range = valid_id_range
        self.lock_min_dwell = float(lock_min_dwell)
        self.lost_grace_sec = float(lost_grace_sec)
        self._lock_since: float = 0.0

        self.gimbal_yaw: Optional[float] = None
        self.gimbal_pitch: Optional[float] = None
        self.gimbal_angle_ts: float = 0.0

        self._tracks = {}
        self._tracks_lock = threading.Lock()
        self._captured_ids = set()
        self._locked_id: Optional[int] = None
        self._last_lock_ts: float = 0.0
        self._cooldown_until: float = 0.0

        self._miss = 0
        self._stream_ready = False

        self._cap_thread: Optional[threading.Thread] = None
        self._cap_active = False
        self._last_save_path: Optional[str] = None
        self._save_dir = os.path.abspath(save_dir)
        os.makedirs(self._save_dir, exist_ok=True)

        self._last_cmd_log_t = 0.0
        self._last_cmd_info = "IDLE"
        self.anti_osc_flip_window = float(anti_osc_flip_window)
        self.anti_osc_max_flips   = int(anti_osc_max_flips)
        self.anti_osc_freeze      = float(anti_osc_freeze)
        self._ao_last_sign        = 0
        self._ao_flip_times: list = []
        self._ao_freeze_until     = 0.0
        self.settle_frames        = int(settle_frames)
        self._settle_cnt          = 0
        self._yaw_cur_dps         = 0.0
        self._yaw_slew_dps        = 12.0

        if self.show_debug:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    def cooldown_for(self, sec: float = 3.0):
        now = time.time()
        self._locked_id = None
        self._last_lock_ts = 0.0
        self._cooldown_until = now + float(sec)
        try:
            self._ao_freeze_until = now + 0.35
        except Exception:
            pass

    def subscribe(self) -> None:
        # marker 检测
        try:
            self.ep_vision.sub_detect_info(name="marker", callback=self._on_detect_marker)
            print("[Marker] sub_detect_info: marker")
        except Exception as e:
            print("[Marker][WARN] subscribe failed:", e)
        try:
            self.ep_robot.gimbal.sub_angle(freq=10, callback=self._on_gimbal_angle)
            print("[Gimbal] sub_angle @10Hz")
        except Exception as e:
            print("[Gimbal][WARN] sub_angle:", e)

    def unsubscribe(self) -> None:
        try:
            self.ep_vision.unsub_detect_info(name="marker")
        except Exception:
            pass
        try:
            self.ep_robot.gimbal.unsub_angle()
        except Exception:
            pass

    def start_stream_with_warmup(self) -> bool:
        ok = self._safe_start_video_stream()
        if not ok:
            ok = self._safe_start_video_stream(resolution=None, bitrate=None, fps=None)
        if not ok:
            print("[Camera][ERR] start stream failed")
            return False
        time.sleep(0.5)
        try:
            _ = self._read_cv2_image()
            self._stream_ready = True
            return True
        except Empty:
            print("[Camera][WARN] warm read timeout; stream still started")
            self._stream_ready = True
            return True

    def _safe_start_video_stream(self, resolution=None, bitrate=None, fps=None) -> bool:
        try:
            if resolution is None:
                self.ep_camera.start_video_stream(display=False); return True
            if (bitrate is not None) and (fps is not None):
                self.ep_camera.start_video_stream(display=False, resolution=resolution, bitrate=bitrate, fps=fps); return True
        except Exception:
            pass
        try:
            if (resolution is not None) and (fps is not None):
                self.ep_camera.start_video_stream(display=False, resolution=resolution, fps=fps); return True
        except Exception:
            pass
        try:
            if (resolution is not None) and (bitrate is not None):
                self.ep_camera.start_video_stream(display=False, resolution=resolution, bitrate=bitrate); return True
        except Exception:
            pass
        try:
            if resolution is not None:
                self.ep_camera.start_video_stream(display=False, resolution=resolution); return True
        except Exception:
            pass
        return False

    def _read_cv2_image(self):
        try:
            return self.ep_camera.read_cv2_image(strategy="newest", timeout=self.read_timeout)
        except Exception:
            return self.ep_camera.read_cv2_image(timeout=self.read_timeout)

    def read_frame(self) -> Optional[any]:
        try:
            img = self._read_cv2_image()
            self._miss = 0
            return img
        except Empty:
            self._miss += 1
            if self._miss in (3, 6):
                time.sleep(0.2)
            if self._miss >= self.miss_restart:
                self._restart_stream()
            return None

    def _restart_stream(self):
        print("[Camera] restarting stream…")
        try:
            self.ep_camera.stop_video_stream()
        except Exception:
            pass
        time.sleep(0.3)
        self.start_stream_with_warmup()
        self._miss = 0

    def _on_detect_marker(self, marker_info):
        now = time.time()
        try:
            for i in range(len(marker_info)):
                x, y, w, h, info = marker_info[i]
                try:
                    mid = int(info)
                except Exception:
                    continue
                if not (self.valid_id_range[0] <= mid <= self.valid_id_range[1]):
                    continue
                if w < self.track_width_min:
                    continue
                with self._tracks_lock:
                    if mid not in self._tracks:
                        self._tracks[mid] = {'x': x, 'y': y, 'w': w, 'h': h,
                                             'ts': now, 'first_ts': now}
                    else:
                        t = self._tracks[mid]
                        a = self.smooth_alpha
                        t['x'] = a * x + (1 - a) * t['x']
                        t['y'] = a * y + (1 - a) * t['y']
                        t['w'] = a * w + (1 - a) * t['w']
                        t['h'] = a * h + (1 - a) * t['h']
                        t['ts'] = now
        except Exception:
            pass

    def _on_gimbal_angle(self, angle_info):
        try:
            if isinstance(angle_info, (list, tuple)) and len(angle_info) >= 2:
                self.gimbal_yaw = float(angle_info[0])
                self.gimbal_pitch = float(angle_info[1])
            elif isinstance(angle_info, dict):
                self.gimbal_yaw = float(angle_info.get("yaw", self.gimbal_yaw))
                self.gimbal_pitch = float(angle_info.get("pitch", self.gimbal_pitch))
            else:
                return
            self.gimbal_angle_ts = time.time()
        except Exception:
            pass


    def _stable_markers(self) -> List[MarkerInfo]:
        now = time.time()
        stable = []
        with self._tracks_lock:
            for mid, t in list(self._tracks.items()):
                if now - t['ts'] <= self.track_hold_sec:
                    stable.append(MarkerInfo(t['x'], t['y'], t['w'], t['h'], mid))
        return stable

    def _choose_target(self, stable: List[MarkerInfo]) -> Optional[MarkerInfo]:
        now = time.time()
        candidates = [m for m in stable if int(m.mid) not in self._captured_ids]
        if self.only_near:
            candidates = [m for m in candidates if m.w >= self.near_width_min]

        if self._locked_id is not None:
            locked_obj = None
            for m in candidates:
                if int(m.mid) == int(self._locked_id):
                    locked_obj = m
                    break

            if locked_obj is not None:
                self._last_lock_ts = now
                if (now - self._lock_since) < self.lock_min_dwell:
                    return locked_obj
                return locked_obj

            if (now - self._last_lock_ts) <= self.lost_grace_sec:
                return None
            self._locked_id = None

        if not candidates:
            return None

        def first_ts_of(mid: int) -> float:
            t = self._tracks.get(int(mid), {})
            return t.get('first_ts', float('inf'))

        m = min(candidates, key=lambda k: first_ts_of(k.mid))
        self._locked_id = int(m.mid)
        self._lock_since = now
        self._last_lock_ts = now
        return m

    def _ready_for_capture(self, target: MarkerInfo) -> bool:
        if time.time() < self._cooldown_until:
            return False
        center_err = target.x - 0.5
        return (abs(center_err) < self.center_tol) and (target.w >= self.width_thresh_norm)

    def _clamp(self, v, lo, hi):
        return max(lo, min(v, hi))

    def _send_yaw(self, yaw_speed_norm: float, max_speed_norm: float = 0.25) -> None:
        now = time.time()
        if now < self._ao_freeze_until:
            try:
                self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            self._log_cmd("AO-FREEZE")
            return

        unit = self._clamp(yaw_speed_norm / max(1e-6, max_speed_norm), -1.0, 1.0)
        yaw_dps_raw = unit * self.max_yaw_dps

        if abs(yaw_dps_raw) < self.deadband_dps:
            target = 0.0
            if target > self._yaw_cur_dps:
                self._yaw_cur_dps = min(self._yaw_cur_dps + self._yaw_slew_dps, target)
            elif target < self._yaw_cur_dps:
                self._yaw_cur_dps = max(self._yaw_cur_dps - self._yaw_slew_dps, target)
            try:
                self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=self._yaw_cur_dps * self.dir_yaw)
            except Exception:
                if self.ep_chassis is not None:
                    self.ep_chassis.drive_speed(x=0, y=0, z=self._yaw_cur_dps * self.dir_yaw, timeout=0.1)
            self._log_cmd("IDLE (deadband)")
            return

        sign = 1 if yaw_dps_raw > 0 else -1
        if sign != self._ao_last_sign and self._ao_last_sign != 0:
            self._ao_flip_times.append(now)
            self._ao_flip_times = [t for t in self._ao_flip_times if now - t <= self.anti_osc_flip_window]
            if len(self._ao_flip_times) >= self.anti_osc_max_flips:
                self._ao_freeze_until = now + self.anti_osc_freeze
                self._ao_flip_times.clear()
                try:
                    self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                except Exception:
                    pass
                self._log_cmd(f"AO-TRIGGER freeze {self.anti_osc_freeze:.2f}s")
                return
        self._ao_last_sign = sign

        if abs(yaw_dps_raw) < self.min_yaw_dps:
            yaw_dps_raw = self.min_yaw_dps * (1 if yaw_dps_raw > 0 else -1)

        target = yaw_dps_raw
        if target > self._yaw_cur_dps:
            self._yaw_cur_dps = min(self._yaw_cur_dps + self._yaw_slew_dps, target)
        else:
            self._yaw_cur_dps = max(self._yaw_cur_dps - self._yaw_slew_dps, target)

        yaw_cmd = self.dir_yaw * self._yaw_cur_dps
        try:
            self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=yaw_cmd)
            info = f"GIMBAL yaw={yaw_cmd:.1f} dps"
        except Exception:
            if self.ep_chassis is not None:
                self.ep_chassis.drive_speed(x=0, y=0, z=yaw_cmd, timeout=0.1)
                info = f"CHASSIS z={yaw_cmd:.1f}"
            else:
                info = "NO CHASSIS"
        self._log_cmd(info)

    def _log_cmd(self, info: str):
        t = time.time()
        if t - self._last_cmd_log_t > 0.5:
            print("[CMD]", info)
            self._last_cmd_log_t = t
        self._last_cmd_info = info

    def _recenter(self, wait: bool = True):
        try:
            self.ep_robot.gimbal.recenter(pitch_speed=self.recenter_pitch_speed,
                                          yaw_speed=self.recenter_yaw_speed)
            print(f"[INFO] gimbal.recenter issued ({self.recenter_yaw_speed} dps)")
        except Exception as e:
            print(f"[WARN] gimbal.recenter not available: {e}")
            try:
                self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            return
        if not wait:
            return
        t0 = time.time()
        while time.time() - t0 < self.recenter_timeout:
            if self.gimbal_yaw is not None and abs(self.gimbal_yaw) <= self.recenter_tol_deg:
                print("[INFO] gimbal recentered")
                return
            time.sleep(0.05)
        print("[WARN] recenter wait timeout")

    def _capture_worker(self, mid: int, get_frame_callable):
        try:
            time.sleep(self.pre_capture_delay)
            _ = get_frame_callable()
            frame = get_frame_callable()
            if frame is None:
                return
            h, w = frame.shape[:2]
            with self._tracks_lock:
                t = self._tracks.get(mid)
                if not t:
                    return
                x, y, bw, bh = t['x'], t['y'], t['w'], t['h']
            x1 = int((x - bw/2) * w); y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w); y2 = int((y + bh/2) * h)
            x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))

            label = f"Team {self.team_id:02d} detects a marker with ID of {mid}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = max(0, min(x1, w - tw - 6)); ty = max(th+6, y1-8)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (tx-4, ty-th-base-4), (tx+tw+4, ty+base+4), (0, 0, 0), -1)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._save_dir, f"captured_marker_{mid}_{ts}.jpg")
            cv2.imwrite(path, frame)
            self._last_save_path = path
            print(f"[SHOT] saved: {path}")

            self._captured_ids.add(int(mid))
            self._cooldown_until = time.time() + self.cooldown_s

            if self.recenter_after_capture:
                try:
                    self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                except Exception:
                    pass
                self._recenter(wait=True)
        finally:
            self._cap_active = False

    def _capture_async(self, mid: int):
        if self._cap_active:
            return
        self._cap_active = True
        self._last_save_path = None
        t = threading.Thread(target=self._capture_worker, args=(mid, self.read_frame), daemon=True)
        self._cap_thread = t
        t.start()

    def _draw_overlays(self, img, stable: List[MarkerInfo], target: Optional[MarkerInfo]):
        h, w = img.shape[:2]
        for m in stable:
            x1 = int((m.x - m.w/2) * w); y1 = int((m.y - m.h/2) * h)
            x2 = int((m.x + m.w/2) * w); y2 = int((m.y + m.h/2) * h)
            x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
            color = (0, 255, 0) if (target and m.mid == target.mid) else (0, 200, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cx, cy = int(m.x * w), int(m.y * h)
            cv2.circle(img, (cx, cy), 4, color, -1)
            label = f"Team 20 detect a marker with ID {m.mid} w={m.w:.2f}"
            cv2.putText(img, label, (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cx = int(0.5 * w); cy = int(0.5 * h)
        cv2.drawMarker(img, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=2)
        state = {
            'lock': self._locked_id,
            'cool': max(0.0, self._cooldown_until - time.time()),
            'lastcmd': self._last_cmd_info,
        }
        cv2.putText(img, f"lock:{state['lock']} cool:{state['cool']:.1f}s {state['lastcmd']}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    def step(self, frame=None, draw=True):
        own_frame = False
        if frame is None:
            frame = self.read_frame()
            own_frame = True
        if frame is None:
            return None, None, 'IDLE', None


        stable = self._stable_markers()
        target = self._choose_target(stable)

        stage = 'IDLE'
        if target is not None:
            center_err = target.x - 0.5
            if abs(center_err) <= (self.center_tol * 0.8):
                try:
                    self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                except Exception:
                    pass
                yaw_speed_norm = 0.0
            else:
                yaw_speed_norm = self._clamp(self.kp_yaw * center_err, -0.25, 0.25)

            self._send_yaw(yaw_speed_norm)

            stage = 'FOLLOW'

            if abs(target.x - 0.5) <= self.center_tol and target.w >= self.width_thresh_norm:
                self._settle_cnt += 1
            else:
                self._settle_cnt = 0

            if self._settle_cnt >= self.settle_frames and self._ready_for_capture(target):
                self._capture_async(int(target.mid))
                self._settle_cnt = 0
                stage = 'CAPTURING'

        else:
            if self._locked_id and time.time() - self._last_lock_ts <= self.track_hold_sec:
                pass
            else:
                self._locked_id = None

        if draw and self.show_debug:
            vis = frame.copy()
            try:
                self._draw_overlays(vis, stable, target)
            except Exception:
                pass
            cv2.imshow(self.window_name, vis)
            cv2.waitKey(1)

        save_path = self._last_save_path
        self._last_save_path = None
        return frame, target, stage, save_path

    def stop(self):
        try:
            self.unsubscribe()
        except Exception:
            pass
        try:
            self.ep_camera.stop_video_stream()
        except Exception:
            pass
        try:
            if self.ep_chassis is not None:
                self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
        except Exception:
            pass
        if self.show_debug:

            cv2.destroyWindow(self.window_name)