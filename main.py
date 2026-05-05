# =========================================================
# Batch Vehicle Detection + Tracking + Feature Extraction
# 输出：
#   1) 视频：outputs/videos/processed_xxx.mp4
#   2) 特征：outputs/features/xxx_features.csv
# =========================================================
import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from bev_error import compute_distance_errors, summarize_errors
from bev_utils import init_bev, generate_bev_frame
from common import _load_calibration_file, _get_default_calibration, \
    detect_zoom_radial_flow
from lane_utils import get_lane_boundaries_from_lines
from roi_utils import  point_in_polygon
from src import Annotator, ViewTransformer, SpeedEstimator
from config import (
    YOLO_MODEL_PATH,
    WINDOW_NAME, OUTPUT_VIDEO_DIR, OUTPUT_FEATURE_DIR, INPUT_DIR, VIDEO_EXTS, COLUMNS_FULL, AUTO_MODE, USE_ROI,
    STABLE_NEEDED, AUTO_ZOOM_DETECTION, SKIP_EXISTING_FEATURES
)
from calibration import interactive_calibration, save_calibration, get_roi_for_video, interactive_calibration_scale


# =========================================================
# 1️⃣ 标定获取函数（修改：支持车道线保存）
# =========================================================
def get_calibration(input_path, video_info=None):
    """
    获取标定数据（透视变换源点、目标点、计数线及车道线）
    返回: (SOURCE, TARGET, line_start, line_end, lane_lines)
    """
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    global_path = "calibration/global.npy"
    video_path = f"calibration/{video_name}.npy"

    has_global = os.path.exists(global_path)
    has_video = os.path.exists(video_path)

    print(f"\n🎯 Calibration for: {video_name}")

    # ---------- 自动模式 ----------
    if AUTO_MODE:
        if has_video:
            print("⚡ AUTO → 使用视频标定")
            return _load_calibration_file(video_path)
        if has_global:
            print("⚡ AUTO → 使用全局标定")
            return _load_calibration_file(global_path)
        print("⚠️ AUTO → 无标定，进入交互标定")
        calib = interactive_calibration(input_path)
        save_calibration(global_path, *calib)
        return calib

    # ---------- 手动模式 ----------
    print("1️⃣ 使用该视频已存储的标定数据" if has_video else "1️⃣（无）")
    print("2️⃣ 使用全局已存储的数据标定" if has_global else "2️⃣（无）")
    print("3️⃣ 重新标定")
    print("4️⃣ 使用测试数据（临时，不保存）")
    choice = input("输入 1 / 2 / 3 / 4：").strip()

    # 选项1：视频标定
    if choice == "1" and has_video:
        return _load_calibration_file(video_path)

    # 选项4：测试数据（无需已有标定文件）
    if choice == "4":
        return _get_default_calibration(video_info)

    # 选项2：全局标定
    if choice == "2" and has_global:
        return _load_calibration_file(global_path)

    # 选项3 或 无效输入：重新标定
    calib = interactive_calibration(input_path)

    # 标定数据保存
    save_choice = input("标定数据保存方式：1=本视频 2=全局 3=全部：").strip()
    is_save_video = save_choice in ("1", "3")
    is_save_global = save_choice in ("2", "3")

    if is_save_video:
        save_calibration(video_path, *calib)
    if is_save_global:
        save_calibration(global_path, *calib)

    return calib


# =========================================================
# 2️⃣ 单视频处理函数（修改：使用手动车道线）
# =========================================================
def process_video(input_path, output_video_path, feature_dir, show=False):
    """
    处理单个视频文件：检测、跟踪、速度估算、特征提取、BEV 生成。
    支持自动检测画面缩放（变焦），可通过 config.AUTO_ZOOM_DETECTION 开关控制。
    """
    print(f"\n🚀 Processing: {input_path}")

    # ---------- 缩放检测与自动重标定状态（仅在开关启用时有效） ----------
    if AUTO_ZOOM_DETECTION:
        calib_version = 0  # 新增：标定版本号，每次重标定后+1
        prev_frame_for_zoom = None          # 上一帧图像（用于缩放检测）
        calibrated = True                   # 当前标定是否有效（True=正常测速，False=变焦导致标定失效）
        stable_counter = 0                  # 连续稳定帧计数器（无缩放时累加，达到阈值触发重标定）
        # 缩放检测频率控制
        ZOOM_CHECK_INTERVAL = 10            # 每 10 帧检测一次缩放
        zoom_check_counter = 0              # 距离上次检测经过的帧数
        last_is_zoomed = False              # 缓存上一次检测结果
        last_scale = 1.0
    else:
        calib_version = -1  # 未启用时标记为 -1
        # 开关关闭时，始终认为标定有效，不进行任何缩放检测
        calibrated = True
        # 定义占位变量（避免后续引用错误，但不会被使用）
        prev_frame_for_zoom = None
        zoom_check_counter = 0

    # 初始化误差数组（用于统计透视误差）
    all_errors = []

    # 获取视频基本信息（宽度、高度、帧率等）
    video_info = sv.VideoInfo.from_video_path(input_path)

    # 创建 ByteTrack 跟踪器（用于跨帧车辆 ID 关联）
    tracker = sv.ByteTrack(frame_rate=video_info.fps)

    # =========================
    # 获取标定数据（透视四点、计数线、车道线）
    # =========================
    SOURCE, TARGET, line_start, line_end, lane_lines = get_calibration(input_path, video_info=video_info)

    # ========== ROI 加载（如果启用） ==========
    roi_pts = None
    if USE_ROI:
        cap_temp = cv2.VideoCapture(input_path)
        ret, first_frame = cap_temp.read()
        cap_temp.release()
        if ret:
            roi_pts = get_roi_for_video(input_path, first_frame)
            if roi_pts is not None:
                print(f"📐 ROI 多边形已加载，顶点数: {len(roi_pts)}")
        else:
            print("⚠️ 无法读取视频第一帧，跳过 ROI 设置")

    ###################### BEV 输出设置 ######################
    bev_config = init_bev(TARGET, pix_per_meter=20)
    src_pts = np.array(SOURCE, dtype=np.float32)
    dst_pts = np.array(TARGET, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    world_width = np.max(dst_pts[:, 0]) - np.min(dst_pts[:, 0])
    world_height = np.max(dst_pts[:, 1]) - np.min(dst_pts[:, 1])
    print(f"🌍 世界范围: {world_width:.2f}m x {world_height:.2f}m")
    PIX_PER_METER = 1
    bev_width = max(100, int(world_width * PIX_PER_METER))
    bev_height = max(100, int(world_height * PIX_PER_METER))
    base, ext = os.path.splitext(output_video_path)
    bev_output_path = f"{base}_bev{ext}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    bev_writer = cv2.VideoWriter(bev_output_path, fourcc, video_info.fps, (bev_width, bev_height))

    # -----------------------------
    # 计数线
    # -----------------------------
    start = sv.Point(line_start[0], line_start[1])
    end = sv.Point(line_end[0], line_end[1])
    line_zone = sv.LineZone(start, end, minimum_crossing_threshold=1)

    # -----------------------------
    # 透视变换器、速度估计器
    # -----------------------------
    view_transformer = ViewTransformer(SOURCE, TARGET)
    speed_estimator = SpeedEstimator(fps=video_info.fps, view_transformer=view_transformer)

    # 车道边界
    if lane_lines:
        lane_boundaries = get_lane_boundaries_from_lines(lane_lines, view_transformer)
        print(f"🛣️ 检测到 {len(lane_lines)} 条车道线，划分 {len(lane_boundaries) - 1} 个车道")
        print(f"   车道边界（世界坐标 X）: {[f'{b:.2f}m' for b in lane_boundaries]}")
    else:
        lane_boundaries = []
        print("⚠️ 未检测到车道线标定，将使用默认车道划分")

    # 标注器
    annotator = Annotator(
        resolution_wh=video_info.resolution_wh,
        box_annotator=True,
        label_annotator=True,
        line_annotator=True,
        multi_class_line_annotator=True,
        trace_annotator=True,
        polygon_zone=np.asarray(roi_pts) if roi_pts is not None else None,
    )

    # 数据缓存
    all_records = []
    prev_speed_dict = {}
    frame_generator = sv.get_video_frames_generator(input_path)

    with sv.VideoSink(output_video_path, video_info) as sink:
        frame_id = 0
        for frame in frame_generator:
            frame_id += 1
            time_sec = frame_id / video_info.fps

            # -------------------------------------------------
            # 缩放检测与自动重标定（仅当开关启用时执行）
            # -------------------------------------------------
            if AUTO_ZOOM_DETECTION and prev_frame_for_zoom is not None:
                # 每隔 ZOOM_CHECK_INTERVAL 帧进行一次实际检测
                if zoom_check_counter % ZOOM_CHECK_INTERVAL == 0:
                    is_zoomed, scale, _ = detect_zoom_radial_flow(prev_frame_for_zoom, frame)
                    last_is_zoomed = is_zoomed
                    last_scale = scale
                else:
                    is_zoomed = last_is_zoomed
                    scale = last_scale

                if is_zoomed:
                    if calibrated:
                        print(f"⚠️ 检测到缩放 (scale={scale:.3f})，暂停速度计算，等待画面稳定...")
                    calibrated = False
                    stable_counter = 0
                else:
                    if not calibrated:
                        stable_counter += 1
                        if stable_counter >= STABLE_NEEDED:
                            print("🔄 画面已稳定，尝试自动重新标定...")
                            try:
                                # 弹出交互窗口，让用户重新点击四点、计数线、车道线
                                # new_src, new_dst, new_ls, new_le, new_lanes = recalibrate_from_frame(frame, video_info)
                                new_src, new_dst, new_ls, new_le, new_lanes = interactive_calibration_scale(
                                    frame,
                                    video_info,
                                    mode=None,
                                    version=None)

                                # ---------- 更新所有依赖标定的对象 ----------
                                SOURCE, TARGET = new_src, new_dst
                                line_start, line_end = new_ls, new_le
                                lane_lines = new_lanes
                                # 重新创建透视变换器和速度估计器
                                view_transformer = ViewTransformer(SOURCE, TARGET)
                                speed_estimator = SpeedEstimator(fps=video_info.fps, view_transformer=view_transformer)
                                # 重新计算车道边界
                                if lane_lines:
                                    lane_boundaries = get_lane_boundaries_from_lines(lane_lines, view_transformer)
                                else:
                                    lane_boundaries = []
                                # 重新创建计数线对象
                                start = sv.Point(line_start[0], line_start[1])
                                end = sv.Point(line_end[0], line_end[1])
                                line_zone = sv.LineZone(start, end, minimum_crossing_threshold=1)
                                # 重置跟踪器和速度缓存
                                tracker = sv.ByteTrack(frame_rate=video_info.fps)
                                prev_speed_dict.clear()
                                # 重置计数（可选）
                                # line_zone.reset()
                                # ---------- 更新 BEV 相关的变换和视频尺寸 ----------
                                src_pts = np.array(SOURCE, dtype=np.float32)
                                dst_pts = np.array(TARGET, dtype=np.float32)
                                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                                world_width = np.max(dst_pts[:, 0]) - np.min(dst_pts[:, 0])
                                world_height = np.max(dst_pts[:, 1]) - np.min(dst_pts[:, 1])
                                bev_width = max(100, int(world_width * PIX_PER_METER))
                                bev_height = max(100, int(world_height * PIX_PER_METER))
                                # 释放旧的 BEV 写入器并重新创建
                                bev_writer.release()
                                bev_writer = cv2.VideoWriter(bev_output_path, fourcc, video_info.fps, (bev_width, bev_height))
                                bev_config = init_bev(TARGET, pix_per_meter=20)
                                print("✅ 重新标定完成，恢复速度计算")
                                calibrated = True
                                stable_counter = 0
                                calib_version += 1
                                print(f"📌 标定版本更新为 {calib_version}")
                            except Exception as e:
                                print(f"❌ 重新标定失败: {e}，继续等待稳定帧或手动干预")
                                stable_counter = 0

                zoom_check_counter += 1
                prev_frame_for_zoom = frame.copy()
            elif AUTO_ZOOM_DETECTION and prev_frame_for_zoom is None:
                # 第一帧：只保存，不检测
                prev_frame_for_zoom = frame.copy()
                zoom_check_counter += 1

            # -------------------------------------------------
            # 如果标定失效（仅当开关开启且 calibrated==False），输出提示帧并跳过正常处理
            # -------------------------------------------------
            if AUTO_ZOOM_DETECTION and not calibrated:
                # 在原始帧上绘制红色警告文字
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, "CALIBRATION LOST - WAITING FOR STABLE SCENE", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                sink.write_frame(annotated_frame)
                # 生成一个纯黑 BEV 帧并绘制警告文字
                bev_frame = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
                cv2.putText(bev_frame, "ZOOM DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                bev_writer.write(bev_frame)
                continue  # 跳过后续所有检测、跟踪、速度计算

            # -------------------------------------------------
            # 正常处理（标定有效时）—— 以下代码与原始版本完全一致
            # -------------------------------------------------

            # ---------------- YOLO 检测 ----------------
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # ---------------- 跟踪（ByteTrack） ----------------
            detections = tracker.update_with_detections(detections)

            # ---------------- 速度估算 ----------------
            detections = speed_estimator.update(detections)

            # ---------- ROI 过滤（如果启用了 ROI，只保留 ROI 内的车辆） ----------
            if roi_pts is not None:
                centers = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                mask = [point_in_polygon(cx, cy, roi_pts) for cx, cy in centers]
                mask = np.array(mask, dtype=bool)
                detections = detections[mask]

            # ---------------- 计数（通过计数线） ----------------
            line_zone.trigger(detections)

            # ---------------- 坐标转换：图像坐标 -> 世界坐标 ----------------
            centers = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            world_points = view_transformer.transform_points(centers)

            # ---------------- 车道划分（基于手动标定的车道线边界） ----------------
            lane_ids = []
            if lane_boundaries and len(lane_boundaries) >= 2:
                for wx, wy in world_points:
                    lane = -1
                    for i in range(len(lane_boundaries) - 1):
                        if lane_boundaries[i] <= wx <= lane_boundaries[i + 1]:
                            lane = i
                            break
                    if lane == -1:
                        if wx < lane_boundaries[0]:
                            lane = 0
                        elif wx > lane_boundaries[-1]:
                            lane = len(lane_boundaries) - 2
                    lane_ids.append(lane)
            else:
                lane_ids = [int(min(2, max(0, wx // 3.5))) for wx, wy in world_points]

            # =========================
            # 误差统计（每帧的车道内车辆距离误差）
            # =========================
            frame_errors = compute_distance_errors(world_points, lane_ids)
            all_errors.extend(frame_errors)

            # ---------------- 前后车关系（同一车道内按 Y 坐标排序） ----------------
            front_ids = [-1] * len(detections)
            rear_ids = [-1] * len(detections)
            lane_groups = {}
            for i, lane in enumerate(lane_ids):
                lane_groups.setdefault(lane, []).append(i)
            for lane, indices in lane_groups.items():
                sorted_idx = sorted(indices, key=lambda i: world_points[i][1])
                for i, idx in enumerate(sorted_idx):
                    if i > 0:
                        front_ids[idx] = detections.tracker_id[sorted_idx[i - 1]]
                    if i < len(sorted_idx) - 1:
                        rear_ids[idx] = detections.tracker_id[sorted_idx[i + 1]]

            # ---------------- 遍历每个检测到的车辆，提取特征并保存记录 ----------------
            for i in range(len(detections)):
                vid = detections.tracker_id[i]
                cls_id = detections.class_id[i]
                cls = model.model.names[int(cls_id)]

                cx, cy = centers[i]
                wx, wy = world_points[i]

                speed = detections.data["speed"][i] if "speed" in detections.data else 0

                prev_speed = prev_speed_dict.get(vid, speed)
                acc = (speed - prev_speed) * (1000 / 3600) * video_info.fps
                prev_speed_dict[vid] = speed

                x1, y1, x2, y2 = detections.xyxy[i]
                length = (y2 - y1) * 0.05
                width  = (x2 - x1) * 0.05

                record = [
                    frame_id, time_sec, vid, cls,
                    cx, cy,
                    wx, wy,
                    length, width,
                    speed, acc,
                    front_ids[i], rear_ids[i],
                    -1, -1, -1, -1,
                    lane_ids[i],
                    calib_version  # 新增
                ]
                all_records.append(record)

            # ---------------- 可视化（原始视频） ----------------
            labels = []
            class_names = model.model.names
            for i in range(len(detections)):
                vid = detections.tracker_id[i]
                cls = class_names[int(detections.class_id[i])]
                spd = detections.data["speed"][i] if "speed" in detections.data else 0
                if spd and spd != 0:
                    text = f"{cls} {spd:.1f}km/h"
                else:
                    text = f"{cls} #{vid}"
                labels.append(text)

            annotated_frame = annotator.annotate(
                frame,
                detections,
                labels=labels,
                line_zones=[line_zone],
                multi_class_zones=[line_zone],
            )

            sink.write_frame(annotated_frame)

            # ---------------- 生成 BEV 帧并写入 BEV 视频 ----------------
            bev_frame = generate_bev_frame(
                frame=frame,
                SOURCE=SOURCE,
                TARGET=TARGET,
                bev_config=bev_config,
                world_points=world_points,
                lane_ids=lane_ids,
                tracker_ids=detections.tracker_id,
                lane_lines=lane_lines
            )
            bev_writer.write(bev_frame)

            # ---------------- 实时显示（可选） ----------------
            if show:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                cv2.imshow("Bird's Eye View", bev_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # 释放 BEV 写入器
    bev_writer.release()
    cv2.destroyWindow("Bird's Eye View")

    # 计算透视误差
    mean_error, rmse = summarize_errors(all_errors)

    # 保存特征 CSV
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    csv_path = os.path.join(feature_dir, f"{video_name}_features.csv")
    df = pd.DataFrame(all_records, columns=COLUMNS_FULL)
    df = df.sort_values(by=["Frame", "Vehicle_ID"])
    df.to_csv(csv_path, index=False)

    print(f"📊 Features saved: {csv_path}")
    print(f"🚗 Count: {line_zone.in_count + line_zone.out_count}")
    print(f"🛣️ BEV video saved: {bev_output_path}")
    print(f"📏 Mean Error: {mean_error:.3f} m")
    print(f"📐 RMSE: {rmse:.3f} m")



# =========================================================
# 3️⃣ 批量处理
# =========================================================
def process_folder(input_dir, output_video_dir, output_feature_dir, show=False, skip_existing=True):
    video_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(VIDEO_EXTS)
    ]

    if len(video_files) == 0:
        print("❌ No videos found!")
        return
    print(f"📂 Found {len(video_files)} videos")

    for video_name in video_files:
        # 构建特征文件路径
        base_name = os.path.splitext(video_name)[0]
        feature_path = os.path.join(output_feature_dir, f"{base_name}_features.csv")

        # 检查是否跳过
        if skip_existing and os.path.exists(feature_path):
            print(f"⏭️ 跳过 {video_name}，特征文件已存在: {feature_path}")
            continue


        input_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_video_dir, f"processed_{video_name}")
        process_video(
            input_path,
            output_path,
            output_feature_dir,
            show=show
        )


# =========================================================
# 4️⃣ 主入口
# =========================================================
if __name__ == "__main__":
    # =========================================================
    # 1️⃣ 加载模型（只加载一次）
    # =========================================================
    model = YOLO(YOLO_MODEL_PATH)

    # 自动创建目录
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    process_folder(INPUT_DIR, OUTPUT_VIDEO_DIR, OUTPUT_FEATURE_DIR, show=True,skip_existing=SKIP_EXISTING_FEATURES)
    cv2.destroyAllWindows()