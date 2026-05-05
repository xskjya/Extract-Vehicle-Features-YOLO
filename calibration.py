# calibration.py（添加车道标定功能）
import os
import cv2
import numpy as np
from config import SOURCE_POINTS, TARGET_POINTS, LINE_Y, OFFSET, WIDTH, HEIGHT, ROI_DIR
from interactiveCalibration_utils import InteractiveCalibrationQurak, InteractiveCalibrationQurakExtendCav, \
    InteractiveCalibration, InteractiveCalibrationExtendCav
from roi_utils import set_roi


# ==================== 原始接口兼容函数 ====================
def interactive_calibration(video_path, mode=None, version=None):
    """
    交互式标定主函数
    参数:
        mode: 'quick' 或 'full'
        version: 'base'（原始）或 'extend'（扩展画布）
    返回: (SOURCE, TARGET, line_start, line_end, lane_lines)
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("无法读取视频")

    # ==================== 模式选择 ====================
    if mode is None:
        print("\n请选择标定方式：")
        print("1️⃣ 点选标定（点击4个点并输入对应世界坐标）")
        print("2️⃣ 完整标定（默认矩形世界）")
        choice = input("请输入 1 或 2：").strip()

        if choice == '1':
            mode = 'quick'
        elif choice == '2':
            mode = 'full'
        else:
            print("❌ 无效输入，默认使用完整标定")
            mode = 'full'

    # ==================== 版本选择（新增） ====================
    if version is None:
        print("\n请选择标定界面版本：")
        print("1️⃣ 原始版本（轻量）")
        print("2️⃣ 扩展画布版本（推荐，支持更多操作）")
        choice = input("请输入 1 或 2：").strip()

        if choice == '1':
            version = 'base'
        elif choice == '2':
            version = 'extend'
        else:
            print("❌ 无效输入，默认使用扩展版本")
            version = 'extend'

    # ==================== 根据模式 + 版本创建类 ====================
    # ---------- quick ----------
    if mode == 'quick':
        src_pts, world_pts = click_points_with_world_coords(frame)
        line_pts = [(OFFSET, LINE_Y), (frame.shape[1] - OFFSET, LINE_Y)]

        if version == 'base':
            calib = InteractiveCalibrationQurak(
                frame,
                src_points=src_pts,
                world_points=world_pts,
                line_points=line_pts
            )
        else:
            calib = InteractiveCalibrationQurakExtendCav(
                frame,
                src_points=src_pts,
                world_points=world_pts,
                line_points=line_pts
            )

    # ---------- full ----------
    else:
        if version == 'base':
            calib = InteractiveCalibration(frame)
        else:
            calib = InteractiveCalibrationExtendCav(frame)

    # ==================== 运行 ====================
    result = calib.run()
    if result is None:
        raise RuntimeError("标定已取消")

    src_points, world_points, line_start, line_end, lane_lines = result

    SOURCE = np.array(src_points, dtype=np.float32)
    TARGET = np.array(world_points, dtype=np.float32)

    return SOURCE, TARGET, line_start, line_end, lane_lines

def interactive_calibration_scale(frame, video_info, mode=None, version=None, auto=False):
    """
    交互式标定主函数（支持手动/自动模式）
    参数:
        frame: 当前帧图像
        video_info: 视频信息（未使用，保留兼容）
        mode: 'quick' 或 'full'，若为 None 则交互选择（auto=False时）
        version: 'base' 或 'extend'，若为 None 则交互选择（auto=False时）
        auto: bool，是否为自动重标定模式（True=跳过菜单，使用默认 full+extend）
    返回: (SOURCE, TARGET, line_start, line_end, lane_lines)
    """
    # ==================== 自动模式：跳过所有交互，直接使用默认配置 ====================
    if auto:
        # 默认使用完整标定 + 扩展画布版本
        mode = 'full' if mode is None else mode
        version = 'extend' if version is None else version
        print("🔄 自动重标定模式：使用完整标定（扩展画布）")
    else:
        # ==================== 手动模式：交互式选择 ====================
        if mode is None:
            print("\n请选择标定方式：")
            print("1️⃣ 点选标定（点击4个点并输入对应世界坐标）")
            print("2️⃣ 完整标定（默认矩形世界）")
            choice = input("请输入 1 或 2：").strip()
            if choice == '1':
                mode = 'quick'
            elif choice == '2':
                mode = 'full'
            else:
                print("❌ 无效输入，默认使用完整标定")
                mode = 'full'

        if version is None:
            print("\n请选择标定界面版本：")
            print("1️⃣ 原始版本（轻量）")
            print("2️⃣ 扩展画布版本（推荐，支持更多操作）")
            choice = input("请输入 1 或 2：").strip()
            if choice == '1':
                version = 'base'
            elif choice == '2':
                version = 'extend'
            else:
                print("❌ 无效输入，默认使用扩展版本")
                version = 'extend'

    # ==================== 根据模式 + 版本创建标定类 ====================
    if mode == 'quick':
        # 快速标定：用户点击4个点并输入世界坐标
        src_pts, world_pts = click_points_with_world_coords(frame)
        # 计数线使用默认位置（可自行调整 OFFSET, LINE_Y）
        line_pts = [(OFFSET, LINE_Y), (frame.shape[1] - OFFSET, LINE_Y)]

        if version == 'base':
            calib = InteractiveCalibrationQurak(
                frame,
                src_points=src_pts,
                world_points=world_pts,
                line_points=line_pts
            )
        else:  # extend
            calib = InteractiveCalibrationQurakExtendCav(
                frame,
                src_points=src_pts,
                world_points=world_pts,
                line_points=line_pts
            )
    else:  # full
        if version == 'base':
            calib = InteractiveCalibration(frame)
        else:  # extend
            calib = InteractiveCalibrationExtendCav(frame)

    # ==================== 运行标定 ====================
    result = calib.run()
    if result is None:
        raise RuntimeError("标定已取消")

    src_points, world_points, line_start, line_end, lane_lines = result
    SOURCE = np.array(src_points, dtype=np.float32)
    TARGET = np.array(world_points, dtype=np.float32)

    return SOURCE, TARGET, line_start, line_end, lane_lines



def click_points_with_world_coords(frame, window_name="Click 4 points"):
    import tkinter as tk
    from tkinter import simpledialog

    clicked_src = []
    clicked_world = []

    # 创建一个隐藏的 tkinter 根窗口，用于弹出对话框
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)   # 置顶

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_src) < 4:
            idx = len(clicked_src) + 1
            print(f"已选点 {idx}/4: 图像坐标 ({x}, {y})")

            # 弹出世界坐标输入框
            world_p = simpledialog.askstring("世界坐标 X和Y,中间用空格分隔", f"点 {idx} 的世界X和Y坐标（米）:", parent=root)
            if world_p is None:
                print("❌ 已取消该点")
                return

            # 分割
            world_x, world_y = float(world_p.split(" ")[0]) ,float(world_p.split(" ")[1])
            clicked_src.append((x, y))
            clicked_world.append((world_x, world_y))
            print(f"   世界坐标: ({world_x:.2f}, {world_y:.2f})")

            # 更新图像显示
            display = frame.copy()
            for i, pt in enumerate(clicked_src):
                cv2.circle(display, pt, 6, (0, 0, 255), -1)
                cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                if i < len(clicked_world):
                    wpos = clicked_world[i]
                    cv2.putText(display, f"({wpos[0]:.1f},{wpos[1]:.1f})", (pt[0]+10, pt[1]+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            cv2.imshow(window_name, display)

    h, w = frame.shape[:2]  # 原始图像高和宽
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)  # 设置为图像原始尺寸
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, frame)
    print("\n📌 快速标定模式：请依次点击4个点")
    print("   每点后自动弹出对话框输入世界坐标（米）")
    print("   完成4个点后按 ENTER 键确认，ESC 取消。")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13:  # ENTER
            if len(clicked_src) == 4:
                break
            else:
                print(f"❌ 还需要 {4 - len(clicked_src)} 个点")
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            root.destroy()
            raise RuntimeError("标定已取消")

    cv2.destroyWindow(window_name)
    root.destroy()
    return clicked_src, clicked_world


def save_calibration(save_path, SOURCE, TARGET, line_start, line_end, lane_lines=None):
    """保存标定数据（含车道线）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if lane_lines is None:
        lane_lines = []
    np.save(save_path, {
        "SOURCE": SOURCE,
        "TARGET": TARGET,
        "LINE": [line_start, line_end],
        "LANE_LINES": lane_lines
    })
    print(f"💾 标定数据已保存至: {save_path}")


def load_calibration(load_path):
    """加载标定数据（含车道线）"""
    data = np.load(load_path, allow_pickle=True).item()
    lane_lines = data.get("LANE_LINES", [])   # 如果没有车道线，返回空列表
    return data["SOURCE"], data["TARGET"], data["LINE"][0], data["LINE"][1], lane_lines

# ==================== ROI 区域设置 ====================
def interactive_roi_selection(frame, window_name="Set ROI"):
    """
    交互式绘制 ROI 多边形（使用新的 set_roi 界面）
    返回多边形顶点列表（list of tuples）或空列表（取消/点不足）
    """
    return set_roi(frame)
def save_roi(roi_path, roi_points):
    """保存 ROI 顶点"""
    os.makedirs(os.path.dirname(roi_path), exist_ok=True)
    np.save(roi_path, roi_points)
    print(f"💾 ROI 已保存至: {roi_path}")
def load_roi(roi_path):
    """加载 ROI 顶点，不存在则返回 None"""
    if not os.path.exists(roi_path):
        return None
    return np.load(roi_path, allow_pickle=True)
def get_roi_for_video(video_path, frame):
    """
    获取视频对应的 ROI
    如果文件存在则加载，否则询问用户是否创建
    返回 ROI 顶点（numpy array）或 None
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    roi_path = os.path.join(ROI_DIR, f"{video_name}_roi.npy")

    if os.path.exists(roi_path):
        print(f"📁 已存在 ROI 文件: {roi_path}")
        user = input("是否使用已存在的 ROI？(Y/n): ").strip().lower()
        if user != 'n':
            return load_roi(roi_path)

    print("🎯 未找到 ROI 或选择重新创建")
    user = input("是否创建 ROI 区域？(y/N): ").strip().lower()
    if user == 'y':
        roi_pts = interactive_roi_selection(frame)
        if roi_pts is not None:
            save_roi(roi_path, roi_pts)
            return roi_pts
    return None



