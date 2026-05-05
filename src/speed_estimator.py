import numpy as np
import supervision as sv
from collections import defaultdict, deque
from .view_transformer import ViewTransformer


class SpeedEstimator:
    """
    Estimates object speed based on movement across frames using perspective transformation.
    """

    def __init__(
        self,
        fps: int,
        view_transformer: ViewTransformer,
        max_history_seconds: int = 1,
    ):
        """
        Args:
            fps (int): Video frames per second.
            view_transformer (ViewTransformer): Instance for perspective transformation.
            max_history_seconds (int): Max time window to calculate speed (in seconds).
        """
        self.fps = fps
        self.view_transformer = view_transformer
        self.coordinates = defaultdict(
            lambda: deque(maxlen=int(fps * max_history_seconds))
        )

    def calculate_speed(self, tracker_id: int) -> float | None:
        """
        Calculate speed for a specific tracker ID.

        Returns:
            float | None: Speed in km/h if enough data, else None.
        """
        coords = self.coordinates[tracker_id]
        if len(coords) > self.fps / 2:  # Ensure enough movement history
            start, end = coords[0], coords[-1]
            # Euclidean distance in transformed space
            distance = np.linalg.norm(end - start)  # in meters

            time = len(coords) / self.fps  # (N/N) * S = S
            speed = (distance / time) * 3.6  # Convert m/s to km/h
            return int(speed)
        return None

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update object positions and compute speed for current frame.

        Args:
            detections (sv.Detections): Current frame detections.

        Returns:
            sv.Detections: Updated detections with 'speed' in data dictionary.
        """
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = self.view_transformer.transform_points(points)

        speeds_for_frame = []
        for tracker_id, point in zip(detections.tracker_id, points):
            self.coordinates[tracker_id].append(point)
            speed = self.calculate_speed(tracker_id)
            speeds_for_frame.append(speed if speed else 0)

        detections.data["speed"] = np.array(speeds_for_frame)
        return detections


class SpeedEstimatorExtKalman:
    def __init__(
        self,
        fps: int,
        view_transformer: ViewTransformer,
        max_history_seconds: int = 1,
        kalman_q: float = 0.1,
        kalman_r: float = 5.0,
        kalman_p_init: float = 10.0,
    ):
        """
        集成卡尔曼滤波处理：降低测量误差
        """
        self.fps = fps
        self.view_transformer = view_transformer
        self.coordinates = defaultdict(
            lambda: deque(maxlen=int(fps * max_history_seconds))
        )
        # 为每个 tracker_id 维护一个卡尔曼滤波器
        self.kalman_filters: dict[int, KalmanFilter1D] = {}
        # 保存滤波参数以便按需创建
        self.kalman_cfg = (kalman_q, kalman_r, kalman_p_init)

    def calculate_speed(self, tracker_id: int) -> float | None:
        coords = self.coordinates[tracker_id]
        if len(coords) > self.fps / 2:
            start, end = coords[0], coords[-1]
            distance = np.linalg.norm(end - start)
            time = len(coords) / self.fps
            speed_raw = (distance / time) * 3.6  # 原始计算值 km/h

            # 卡尔曼滤波平滑
            if tracker_id not in self.kalman_filters:
                self.kalman_filters[tracker_id] = KalmanFilter1D(*self.kalman_cfg)
            kf = self.kalman_filters[tracker_id]
            speed_filtered = kf.update(speed_raw)
            return speed_filtered
        return None

    def update(self, detections: sv.Detections) -> sv.Detections:
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = self.view_transformer.transform_points(points)

        speeds_for_frame = []
        for tracker_id, point in zip(detections.tracker_id, points):
            self.coordinates[tracker_id].append(point)
            speed = self.calculate_speed(tracker_id)
            speeds_for_frame.append(speed if speed is not None else 0)

        # 清理已消失目标的坐标缓存和卡尔曼滤波器，防止内存泄漏
        active_ids = set(detections.tracker_id)
        for tid in list(self.coordinates.keys()):
            if tid not in active_ids:
                del self.coordinates[tid]
                self.kalman_filters.pop(tid, None)

        detections.data["speed"] = np.array(speeds_for_frame)
        return detections

class KalmanFilter1D:
    """一维卡尔曼滤波器，用于平滑速度序列"""

    def __init__(self, q: float = 0.1, r: float = 5.0, p_init: float = 10.0):
        self.q = q          # 过程噪声协方差
        self.r = r          # 测量噪声协方差
        self.p = p_init     # 估计误差协方差
        self.x = None       # 状态估计值（速度）
        self.initialized = False

    def update(self, measurement: float) -> float:
        """
        输入测量值，返回滤波后的估计值
        """
        if not self.initialized:
            # 首次测量直接初始化状态
            self.x = measurement
            self.initialized = True
            return self.x

        # 预测步骤：假设速度恒定（状态转移系数为1）
        p_pred = self.p + self.q

        # 更新步骤：计算卡尔曼增益，融合观测
        k = p_pred / (p_pred + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred

        return self.x

    def reset(self):
        """跟踪目标消失时重置滤波器"""
        self.initialized = False
        self.x = None
        self.p = 1.0   # 重置为较大的初始值

