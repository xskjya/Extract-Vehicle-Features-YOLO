import cv2
import numpy as np


def _load_calibration_file(file_path):
    """加载 .npy 标定文件，返回 (SOURCE, TARGET, line_start, line_end, lane_lines)"""
    data = np.load(file_path, allow_pickle=True).item()
    return (data["SOURCE"], data["TARGET"],
            data["LINE"][0], data["LINE"][1],
            data.get("LANE_LINES", []))


def _get_default_calibration(video_info):
    """生成默认测试标定参数"""
    LINE_Y = 480
    SOURCE_POINTS = [[450, 300], [860, 300], [1900, 720], [-660, 720]]
    WIDTH, HEIGHT = 25, 100
    TARGET_POINTS = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]
    offset = 55
    # 若 video_info 为 None，默认使用 1920 宽度
    width = video_info.width if video_info is not None else 1920
    start = (offset, LINE_Y)
    end = (width - offset, LINE_Y)
    return (np.array(SOURCE_POINTS), np.array(TARGET_POINTS),
            start, end, [])  # lane_lines 为空


def add_bev_grid(bev_frame, world_width, world_height, pix_per_meter):
    """在 BEV 图上添加网格线和米制坐标"""
    h, w = bev_frame.shape[:2]
    grid_spacing_m = max(1, int(min(world_width, world_height) / 5))
    grid_spacing_px = grid_spacing_m * pix_per_meter

    # 绘制网格线
    for x in range(0, w, grid_spacing_px):
        cv2.line(bev_frame, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bev_frame, f"{int(x/pix_per_meter)}m", (x+2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    for y in range(0, h, grid_spacing_px):
        cv2.line(bev_frame, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bev_frame, f"{int(y/pix_per_meter)}m", (2, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    return bev_frame

def draw_lanes_on_bev(bev_frame, lane_lines, M, bev_width, bev_height, world_width, world_height):
    """将图像坐标系的车道线投影到 BEV 图上"""
    for (x1, y1, x2, y2) in lane_lines:
        pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1,1,2)
        world_pts = cv2.perspectiveTransform(pts, M)
        if len(world_pts) >= 2:
            # 世界坐标 -> BEV 图像坐标
            px = int(world_pts[0][0][0] / world_width * bev_width)
            py = int(world_pts[0][0][1] / world_height * bev_height)
            qx = int(world_pts[1][0][0] / world_width * bev_width)
            qy = int(world_pts[1][0][1] / world_height * bev_height)
            cv2.line(bev_frame, (px, py), (qx, qy), (0, 255, 255), 2)
    return bev_frame


# =========================================================
# 缩放检测 + 自动重新标定（基于当前帧）
# =========================================================
def detect_zoom_radial_flow(frame1, frame2,
                            max_corners=500,
                            quality_level=0.01,
                            min_distance=10,
                            flow_win_size=(21, 21),
                            max_level=3,
                            ransac_thresh=1.5,
                            min_inlier_ratio=0.3,
                            scale_threshold=0.02):
    """
    基于光流径向模式检测两帧之间是否存在缩放（变焦）。

    参数:
        frame1, frame2: 输入两帧图像 (BGR 顺序)
        max_corners:     最多提取的特征点数量
        quality_level:   Shi-Tomasi 角点质量阈值
        min_distance:    特征点之间的最小像素距离
        flow_win_size:   光流计算窗口大小
        max_level:       光流金字塔层数
        ransac_thresh:   RANSAC 拟合残差阈值（像素）
        min_inlier_ratio: 内点比例阈值，低于此值认为无缩放
        scale_threshold:  判定为缩放的最小 |scale-1| 值

    返回:
        is_zoomed (bool):   是否检测到缩放
        scale (float):      估计的缩放因子（>1 表示放大，<1 表示缩小）
        center (tuple):     缩放中心坐标 (cx, cy) 或 (None, None)
    """
    # 1. 转为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 2. 检测第一帧的特征点（稀疏光流用）
    pts1 = cv2.goodFeaturesToTrack(gray1, maxCorners=max_corners,
                                   qualityLevel=quality_level,
                                   minDistance=min_distance)
    if pts1 is None or len(pts1) < 3:
        return False, 1.0, (None, None)
    pts1 = pts1.reshape(-1, 2).astype(np.float32)

    # 3. 计算光流
    pts2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None,
                                               winSize=flow_win_size,
                                               maxLevel=max_level,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # 保留有效跟踪点
    good = status.ravel() == 1
    if np.sum(good) < 3:
        return False, 1.0, (None, None)
    pts1 = pts1[good]
    pts2 = pts2[good]

    # 4. 计算每个点的运动向量 (vx, vy)
    vx = pts2[:, 0] - pts1[:, 0]
    vy = pts2[:, 1] - pts1[:, 1]

    # 5. RANSAC 拟合缩放+平移模型
    # 模型: vx = a * (x - cx) + tx
    #       vy = a * (y - cy) + ty
    # 其中 a = s-1, s 为缩放因子, (cx, cy) 为缩放中心, (tx, ty) 为平移分量
    # 但这样有 4 个未知数 (a, cx, cy, tx, ty? 实际是5个：a, cx, cy, tx, ty)
    # 更好的做法：用线性形式拟合 a, cx, cy, tx, ty 需要非线性，可先假设平移很小或剔除均值。
    # 为简化且保持鲁棒，我们使用基于“径向运动 + 平移”的线性最小二乘：
    # 令未知数 = [a, b, c] 满足 vx = a*x + b,  vy = a*y + c，然后推导缩放中心。
    # 推导：若 v = (s-1)*(p - c)，则 vx = (s-1)*x - (s-1)*cx，vy = (s-1)*y - (s-1)*cy。
    # 所以令 a = s-1, b = -a*cx, c = -a*cy。则模型为 vx = a*x + b, vy = a*y + c。
    # 注意这里的 a 为全局缩放系数，b,c 与缩放中心相关。
    # 该模型仅有 3 个参数，可以用线性最小二乘直接求解。

    # 构造方程组：对于每个点 i，有：
    #   vx_i = a * x_i + b
    #   vy_i = a * y_i + c
    # 写成矩阵形式： [[x_i, 1, 0], [y_i, 0, 1]] * [a, b, c]^T = [vx_i, vy_i]^T
    A = []
    B = []
    for i in range(len(pts1)):
        x, y = pts1[i]
        A.append([x, 1, 0])
        B.append(vx[i])
        A.append([y, 0, 1])
        B.append(vy[i])
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)

    # 使用 RANSAC 提高鲁棒性
    best_inlier_count = 0
    best_params = None
    best_inlier_mask = None

    # RANSAC 迭代次数（与特征点数量适配）
    n_iter = 500
    if len(pts1) < 50:
        n_iter = 300

    for _ in range(n_iter):
        # 随机选 3 个点（最小样本数）
        idx = np.random.choice(len(pts1), 3, replace=False)
        A_sub = []
        B_sub = []
        for i in idx:
            x, y = pts1[i]
            A_sub.append([x, 1, 0])
            B_sub.append(vx[i])
            A_sub.append([y, 0, 1])
            B_sub.append(vy[i])
        A_sub = np.array(A_sub, dtype=np.float32)
        B_sub = np.array(B_sub, dtype=np.float32)
        try:
            params = np.linalg.lstsq(A_sub, B_sub, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        a, b, c = params
        # 计算所有点的残差
        residuals = []
        for i in range(len(pts1)):
            x, y = pts1[i]
            pred_vx = a * x + b
            pred_vy = a * y + c
            res = np.hypot(vx[i] - pred_vx, vy[i] - pred_vy)
            residuals.append(res)
        residuals = np.array(residuals)
        inliers = residuals < ransac_thresh
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_params = params
            best_inlier_mask = inliers

    if best_params is None or best_inlier_count < max(3, int(min_inlier_ratio * len(pts1))):
        return False, 1.0, (None, None)

    a, b, c = best_params
    # 计算缩放因子 s = a + 1
    s = a + 1.0

    # 计算缩放中心： cx = -b / a, cy = -c / a (a 不能为 0)
    if abs(a) < 1e-3:
        # 缩放因子接近1，实际上没有缩放
        return False, 1.0, (None, None)
    cx = -b / a
    cy = -c / a

    # 判断是否为有效缩放
    is_zoomed = abs(s - 1.0) > scale_threshold

    # 可选：检查内点比例，提升可靠性
    inlier_ratio = best_inlier_count / len(pts1)
    if inlier_ratio < min_inlier_ratio:
        is_zoomed = False

    return is_zoomed, s, (int(cx), int(cy))
