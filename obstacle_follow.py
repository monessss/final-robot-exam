# obstacle_follow.py
# -*- coding: utf-8 -*-
import time
import cv2
import os
import threading

class ObstacleFollower:
    def __init__(self,
                 hold_sec=1.0,
                 slow_h_frac=0.18,
                 stop_h_frac=0.50,
                 follow_speed=0.12,
                 lateral_half_frac=0.20):
        """
        hold_sec: 最近一次识别的“有效期”（秒），超过则视为无目标
        slow_h_frac: 框高占比≥该值 → 减速
        stop_h_frac: 框高占比≥该值 → 停车（建议现场标定≈0.5m对应该阈值）
        follow_speed: 减速时前进速度
        lateral_half_frac: 仅在 |cx-0.5|≤该值 的中心ROI内有效
        """
        self.hold_sec = float(hold_sec)
        self.slow_h_frac = float(slow_h_frac)
        self.stop_h_frac = float(stop_h_frac)
        self.follow_speed = float(follow_speed)
        self.lateral_half_frac = float(lateral_half_frac)

        self.last_box = None     # (x, y, w, h)，像素或归一化
        self.last_ts = 0.0
        self.which_detector = None  # 'robot'/'car'/'people'

    # ---------- 订阅/回调 ----------
    def subscribe(self, ep_vision):
        for name in ("robot", "car", "people"):
            try:
                ep_vision.sub_detect_info(name=name, callback=self._vision_cb)
                try:
                    if name == "robot" and hasattr(ep_vision, "robot_detection"):
                        ep_vision.robot_detection(True)
                    if name == "car" and hasattr(ep_vision, "car_detection"):
                        ep_vision.car_detection(True)
                    if name == "people" and hasattr(ep_vision, "people_detection"):
                        ep_vision.people_detection(True)
                except Exception:
                    pass
                self.which_detector = name
                print(f"[ObstacleFollower] subscribed: {name}")
                return True
            except Exception:
                continue
        print("[ObstacleFollower] subscribe failed (no robot/car/people)")
        return False

    def _vision_cb(self, msg):
        boxes = self._parse_boxes(msg)
        if boxes:
            self.last_box = max(boxes, key=lambda b: b[2] * b[3])
            self.last_ts = time.time()

    @staticmethod
    def _parse_boxes(msg):
        boxes = []
        try:
            items = msg if isinstance(msg, (list, tuple)) else [msg]
            for it in items:
                if isinstance(it, dict):
                    x = float(it.get("x", it.get("bbox_x", 0)))
                    y = float(it.get("y", it.get("bbox_y", 0)))
                    w = float(it.get("w", it.get("bbox_w", 0)))
                    h = float(it.get("h", it.get("bbox_h", 0)))
                elif isinstance(it, (list, tuple)) and len(it) >= 4:
                    x, y, w, h = [float(v) for v in it[:4]]
                else:
                    continue
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
        except Exception:
            pass
        return boxes

    # ---------- 速度融合（仅判阶段/限速/停车 + 可视化；不再拍照） ----------
    def apply(self, x_speed, frame, draw=True):
        H, W = frame.shape[:2]
        active = (time.time() - self.last_ts) <= self.hold_sec
        x_cmd = float(x_speed)

        stage = "NORMAL"  # NORMAL / SLOW / STOP
        color = (0, 255, 0)

        if draw and self.lateral_half_frac > 0:
            x1 = int((0.5 - self.lateral_half_frac) * W)
            x2 = int((0.5 + self.lateral_half_frac) * W)
            cv2.line(frame, (x1, 0), (x1, H), (128, 128, 128), 1)
            cv2.line(frame, (x2, 0), (x2, H), (128, 128, 128), 1)

        if active and self.last_box is not None:
            x, y, w, h = self.last_box
            if h <= 1.5:
                cx_frac = x + w / 2.0
                hf = h
                x_px, y_px, w_px, h_px = int(x * W), int(y * H), int(w * W), int(h * H)
            else:
                cx_frac = (x + w / 2.0) / float(W)
                hf = h / float(H)
                x_px, y_px, w_px, h_px = int(x), int(y), int(w), int(h)

            if self.lateral_half_frac > 0 and abs(cx_frac - 0.5) > self.lateral_half_frac:
                if draw:
                    cv2.putText(frame, "OUT OF ROI", (max(5, x_px), max(20, y_px - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                return x_cmd

            if hf >= self.stop_h_frac:
                stage = "STOP"
                x_cmd = 0.0
                color = (0, 0, 255)
            elif hf >= self.slow_h_frac:
                stage = "SLOW"
                x_cmd = self.follow_speed
                color = (0, 255, 255)
            else:
                stage = "NORMAL"
                color = (0, 255, 0)

            if draw:
                cv2.rectangle(frame, (x_px, y_px), (x_px + w_px, y_px + h_px), color, 3)
                cv2.putText(frame,
                            f"{self.which_detector or 'vision'} "
                            f"cx={cx_frac:.2f} hf={hf:.2f} [{stage}]",
                            (max(5, x_px), max(20, y_px - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return x_cmd
