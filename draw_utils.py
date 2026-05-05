import cv2
import numpy as np


def draw_polyline_dashed(img, points, color, thickness, dash_length=10, gap_length=10):
    """
    在图像上绘制虚线多边形（折线）
    :param img:            画布
    :param points:         点集 (N,2) 或 (N,1,2) 形状
    :param color:          BGR 颜色，如黄色 (0,255,255)
    :param thickness:      线宽
    :param dash_length:    实线段的像素长度
    :param gap_length:     空白段的像素长度
    """
    if len(points) < 2:
        return

    # 确保 points 是 (N,2) 形状
    pts = np.array(points).reshape(-1, 2).astype(np.int32)

    # 对每条线段进行虚线绘制
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        draw_dashed_line(img, p1, p2, color, thickness, dash_length, gap_length)
    # 如果多边形是闭合的，还需要绘制最后一条边（首尾相连）
    # 这里根据调用时的闭合标志决定，本函数默认给定 points 已包含闭合所需的最后一个点或单独处理
    # 简单起见，可以在外部确保 points 是闭环的（首尾点相同或手动闭合）


def draw_dashed_line(img, p1, p2, color, thickness, dash_length, gap_length):
    """
    绘制单条虚线线段
    """
    # 计算线段总长度和方向
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length == 0:
        return
    # 单位方向向量
    ux, uy = dx / length, dy / length

    # 当前累计距离
    current = 0
    # 是否绘制实线（True 表示实线，False 表示空白）
    draw = True
    while current < length:
        if draw:
            # 实线段终点位置
            end = min(current + dash_length, length)
            x1 = int(p1[0] + current * ux)
            y1 = int(p1[1] + current * uy)
            x2 = int(p1[0] + end * ux)
            y2 = int(p1[1] + end * uy)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        # 移动当前距离
        step = dash_length if draw else gap_length
        current += step
        draw = not draw
