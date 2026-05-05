import cv2
import numpy as np


# =========================================================
# 辅助函数：车道判断（基于手动标定的车道线）
# =========================================================
def assign_lane_by_lines(world_x, lane_lines, world_points_x):
    """
    根据手动标定的车道线判断车辆所在车道

    参数:
        world_x: 车辆的世界坐标 X 值
        lane_lines: 车道边界线列表（世界坐标系中的直线）
        world_points_x: 所有车道边界线的 X 坐标列表

    返回:
        lane_id: 车道编号（从0开始），如果无法判断则返回 -1
    """
    if not world_points_x:
        return -1

    # 对车道边界线 X 坐标排序
    boundaries = sorted(world_points_x)

    # 判断车辆 X 坐标在哪个区间
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= world_x <= boundaries[i + 1]:
            return i

    # 超出范围的情况
    if world_x < boundaries[0]:
        return 0
    if world_x > boundaries[-1]:
        return len(boundaries) - 2

    return -1

def get_lane_boundaries_from_lines(lane_lines, view_transformer):
    """
    将图像坐标系的车道线转换为世界坐标系的 X 坐标边界

    参数:
        lane_lines: 图像上的车道线列表 [(x1,y1,x2,y2), ...]
        view_transformer: 透视变换器

    返回:
        boundaries_x: 世界坐标系中所有车道线的 X 坐标列表（排序后）
    """
    boundaries_x = []

    for (x1, y1, x2, y2) in lane_lines:
        # 取线的中点（或两个端点），转换到世界坐标
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # 转换到世界坐标
        points_img = np.array([[mid_x, mid_y]], dtype=np.float32)
        world_points = view_transformer.transform_points(points_img)

        if len(world_points) > 0:
            boundaries_x.append(world_points[0][0])

    return sorted(boundaries_x)


def draw_lanes_on_frame(frame, lane_lines):
    """在图像上绘制手动标定的车道线"""
    for i, (x1, y1, x2, y2) in enumerate(lane_lines):
        # 车道线用绿色粗线
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 绘制车道编号
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # 添加半透明背景
        text = f"Lane {i + 1}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (mid_x - 5, mid_y - text_h - 5),
                      (mid_x + text_w + 5, mid_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
