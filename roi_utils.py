# ==================== 新的 ROI 交互界面 ====================
# 全局变量（仅在 ROI 绘制期间使用）
import cv2
import numpy as np

_roi_points = []               # ROI 多边形顶点列表
_drawing = True                # ROI 绘制中标志
_roi_dragging = False          # 是否拖拽 ROI 顶点
_roi_selected_idx = -1         # 当前选中的 ROI 顶点索引


def _draw_roi_points_and_polygon(img, points, highlight_idx=-1):
    """绘制 ROI 点、多边形连线，并高亮当前选中的点"""
    if not points:
        return img

    for i, (x, y) in enumerate(points):
        if i == highlight_idx:
            radius = max(5, int(img.shape[0] / 100))
            color = (0, 255, 255)   # 黄色高亮
        else:
            radius = max(3, int(img.shape[0] / 150))
            color = (0, 0, 255)     # 红色普通点
        cv2.circle(img, (x, y), radius, color, -1)

    # 顺序连线
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (0, 255, 0), 2)
    # 闭合多边形（至少3个点）
    if len(points) >= 3:
        cv2.line(img, points[-1], points[0], (0, 255, 0), 2)
    return img


def _draw_roi_instructions(img, num_points, drag_mode=False, selected_idx=-1):
    """在图像顶部绘制 ROI 交互提示信息"""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    text1 = "Left click: add point.  Click point to select.  Delete/D key: remove selected point.  Space: finish"
    cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if drag_mode:
        status = "Dragging point - release left button"
        cv2.putText(img, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    else:
        status = f"Points: {num_points}"
        if selected_idx != -1:
            status += f"  |  Selected point: {selected_idx+1} (Press Delete to remove)"
        cv2.putText(img, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    return img


def _draw_roi_callback(event, x, y, flags, param):
    """鼠标回调函数（用于 ROI 绘制）"""
    global _roi_points, _drawing, _roi_dragging, _roi_selected_idx

    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查是否点击到已有顶点
        threshold = 12
        closest_idx = -1
        min_dist = threshold
        for i, (px, py) in enumerate(_roi_points):
            dist = np.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        if closest_idx != -1:
            _roi_selected_idx = closest_idx
            _roi_dragging = True
            print(f"  选中点 {closest_idx+1}，可拖动或按 Delete 键删除")
        else:
            _roi_selected_idx = -1
            _roi_points.append((x, y))
            print(f"  添加ROI点 {len(_roi_points)}: ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE and _roi_dragging and _roi_selected_idx != -1:
        _roi_points[_roi_selected_idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if _roi_dragging:
            _roi_dragging = False
            print(f"  点 {_roi_selected_idx+1} 已移动，仍处于选中状态")

    elif event == cv2.EVENT_RBUTTONDOWN:
        _drawing = False
        print("  右键点击，ROI设置结束")


def set_roi(img):
    """
    手动设置感兴趣区域（ROI）

    功能说明：
        - 左键空白区域：添加 ROI 顶点
        - 单击已有顶点：选中顶点（高亮显示）
        - 拖拽选中顶点：移动位置
        - Delete / D 键：删除选中顶点
        - 空格键：结束 ROI 设置
        - 自动生成封闭多边形 ROI
    """
    global _roi_points, _drawing, _roi_dragging, _roi_selected_idx

    # 重置全局状态
    _roi_points = []
    _drawing = True
    _roi_dragging = False
    _roi_selected_idx = -1

    base_img = img.copy()
    h, w = img.shape[:2]

    # 控制台提示
    print("\n=== 设置感兴趣区域（ROI）===")
    print("操作说明：")
    print("  - 左键点击空白区域：添加顶点")
    print("  - 单击已有顶点：选中并高亮（可拖拽移动）")
    print("  - 拖拽已选中的顶点：移动位置")
    print("  - Delete / D 键：删除选中顶点")
    print("  - 空格键：完成绘制")
    print("提示：至少需要3个顶点才能构成有效多边形\n")

    # 创建窗口并设置缩放
    cv2.namedWindow("Set ROI", cv2.WINDOW_NORMAL)
    max_width, max_height = 1000, 800
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    cv2.resizeWindow("Set ROI", new_w, new_h)

    # 居中窗口
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        x = (screen_width - new_w) // 2
        y = (screen_height - new_h) // 2
        cv2.moveWindow("Set ROI", x, y)
    except Exception as e:
        cv2.moveWindow("Set ROI", 100, 100)
        print(f"窗口居中失败（tkinter 错误: {e}），已使用默认位置")

    cv2.setMouseCallback("Set ROI", _draw_roi_callback)

    # 主循环
    while _drawing:
        disp = base_img.copy()
        disp = _draw_roi_points_and_polygon(disp, _roi_points, _roi_selected_idx)
        disp = _draw_roi_instructions(disp, len(_roi_points), _roi_dragging, _roi_selected_idx)
        cv2.imshow("Set ROI", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):          # 空格完成
            break
        elif key == 127 or key == ord('d') or key == ord('D'):  # 删除键
            if _roi_selected_idx != -1 and len(_roi_points) > 0:
                deleted = _roi_points.pop(_roi_selected_idx)
                print(f"  删除点 {_roi_selected_idx+1}: {deleted}")
                _roi_selected_idx = -1
                _roi_dragging = False
                if len(_roi_points) < 3:
                    print("  当前点数少于3，多边形未闭合，请继续添加点（至少3个）")

    cv2.destroyAllWindows()

    if len(_roi_points) >= 3:
        print(f"\nROI 设置完成，共选定 {len(_roi_points)} 个顶点：{_roi_points}\n")
    else:
        print(f"\n警告：ROI 仅有 {len(_roi_points)} 个点，需至少3个点才能构成有效区域，程序将使用全图\n")

    return _roi_points


# 自定义多边形包含函数（放在文件顶部）
def point_in_polygon(px, py, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        if (y1 > py) != (y2 > py):
            x_intersect = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_intersect:
                inside = not inside
    return inside