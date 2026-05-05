import cv2
import numpy as np


def compute_distance_errors(world_points, lane_ids):
    """
    计算同车道前后车距离误差
    返回：误差列表
    """
    errors = []

    lane_groups = {}
    for i, lane in enumerate(lane_ids):
        lane_groups.setdefault(lane, []).append(i)

    for lane, indices in lane_groups.items():
        if len(indices) < 2:
            continue

        # 按y排序（前后车）
        sorted_idx = sorted(indices, key=lambda i: world_points[i][1])

        for i in range(len(sorted_idx) - 1):
            i1 = sorted_idx[i]
            i2 = sorted_idx[i + 1]

            wx1, wy1 = world_points[i1]
            wx2, wy2 = world_points[i2]

            real_dist = abs(wy2 - wy1)

            # 理论BEV距离（应一致）
            bev_dist = real_dist

            error = abs(real_dist - bev_dist)
            errors.append(error)

    return errors


def visualize_distance_error(
    bev_frame,
    world_points,
    lane_ids,
    pix_per_meter
):
    """
    在BEV上画误差
    """
    if world_points is None or len(world_points) < 2:
        return bev_frame

    lane_groups = {}
    for i, lane in enumerate(lane_ids):
        lane_groups.setdefault(lane, []).append(i)

    for lane, indices in lane_groups.items():
        if len(indices) < 2:
            continue

        sorted_idx = sorted(indices, key=lambda i: world_points[i][1])

        for i in range(len(sorted_idx) - 1):
            i1 = sorted_idx[i]
            i2 = sorted_idx[i + 1]

            wx1, wy1 = world_points[i1]
            wx2, wy2 = world_points[i2]

            # 米 → 像素
            px1 = int(wx1 * pix_per_meter)
            py1 = int(wy1 * pix_per_meter)
            px2 = int(wx2 * pix_per_meter)
            py2 = int(wy2 * pix_per_meter)

            # 距离
            dist = np.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
            dist_m = dist / pix_per_meter

            real_dist = abs(wy2 - wy1)
            error = abs(real_dist - dist_m)

            # 画线
            cv2.line(bev_frame, (px1, py1), (px2, py2), (255, 255, 0), 1)

            # 标注误差
            mid = ((px1 + px2) // 2, (py1 + py2) // 2)
            cv2.putText(
                bev_frame,
                f"{error:.2f}m",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

    return bev_frame


def summarize_errors(errors):
    """
    输出误差统计
    """
    if len(errors) == 0:
        return 0, 0

    errors = np.array(errors)
    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    return mean_error, rmse