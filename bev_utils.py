from bev_error import visualize_distance_error
from common import add_bev_grid, draw_lanes_on_bev


def init_bev(TARGET, pix_per_meter=20):
    import numpy as np

    dst_pts = np.array(TARGET, dtype=np.float32)

    world_width = np.max(dst_pts[:, 0]) - np.min(dst_pts[:, 0])
    world_height = np.max(dst_pts[:, 1]) - np.min(dst_pts[:, 1])

    bev_width = max(34, int(world_width * pix_per_meter))
    bev_height = max(105, int(world_height * pix_per_meter))

    return {
        "world_width": world_width,
        "world_height": world_height,
        "bev_width": bev_width,
        "bev_height": bev_height,
        "pix_per_meter": pix_per_meter
    }


def generate_bev_frame(
    frame,
    SOURCE,
    TARGET,
    bev_config,
    world_points=None,
    lane_ids=None,
    tracker_ids=None,
    lane_lines=None,
):
    import cv2
    import numpy as np

    bev_width = bev_config["bev_width"]
    bev_height = bev_config["bev_height"]
    world_width = bev_config["world_width"]
    world_height = bev_config["world_height"]
    pix_per_meter = bev_config["pix_per_meter"]

    # =========================
    # ✅ 构造正确的透视矩阵
    # =========================
    src = np.array(SOURCE, dtype=np.float32)

    dst_pixels = np.array([
        [0, 0],
        [bev_width, 0],
        [bev_width, bev_height],
        [0, bev_height]
    ], dtype=np.float32)

    M_bev = cv2.getPerspectiveTransform(src, dst_pixels)

    # =========================
    # BEV生成（关键）
    # =========================
    bev_frame = cv2.warpPerspective(frame, M_bev, (bev_width, bev_height))

    # =========================
    # 网格
    # =========================
    bev_frame = add_bev_grid(
        bev_frame,
        world_width,
        world_height,
        pix_per_meter
    )

    # =========================
    # 车道（注意：这里也要用 M_bev）
    # =========================
    if lane_lines:
        bev_frame = draw_lanes_on_bev(
            bev_frame,
            lane_lines,
            M_bev,   # ⭐必须改！
            bev_width,
            bev_height,
            world_width,
            world_height
        )

    # =========================
    # 车辆点（世界→像素）
    # =========================
    if world_points is not None:
        for i, (wx, wy) in enumerate(world_points):

            if not (0 <= wx <= world_width and 0 <= wy <= world_height):
                continue

            px = int(wx * pix_per_meter)
            py = int(wy * pix_per_meter)

            color = (0, 255, 255)
            if lane_ids:
                if lane_ids[i] == 1:
                    color = (0, 165, 255)
                elif lane_ids[i] == 2:
                    color = (0, 0, 255)

            cv2.circle(bev_frame, (px, py), 4, color, -1)

            if tracker_ids is not None:
                cv2.putText(
                    bev_frame,
                    str(tracker_ids[i]),
                    (px + 5, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

    # =========================
    # ✅ 加入误差可视化
    # =========================
    bev_frame = visualize_distance_error(
        bev_frame,
        world_points,
        lane_ids,
        pix_per_meter
    )

    return bev_frame

