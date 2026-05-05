import cv2
import numpy as np
from config import SOURCE_POINTS, TARGET_POINTS, LINE_Y, OFFSET, WIDTH, HEIGHT
from draw_utils import draw_polyline_dashed


######################拓展版本： 最新且推荐##############################################
class InteractiveCalibrationQurakExtendCav:
    def __init__(self, frame, src_points=None, world_points=None, line_points=None):
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.line_thickness = 1  # 统一线条粗细

        # 标定点
        if src_points is not None and world_points is not None:
            self.src_points = list(src_points)
            self.world_points = list(world_points)
        else:
            self.src_points = SOURCE_POINTS
            self.world_points = TARGET_POINTS

        # 计数线
        if line_points is not None:
            self.line_points = list(line_points)
        else:
            self.line_points = [(OFFSET, LINE_Y), (self.w - OFFSET, LINE_Y)]

        # 车道线等
        self.lane_lines = []
        self.lane_calib_mode = False
        self.dragging_idx = -1
        self.active_mode = "src"
        self.hover_idx = -1
        self.mouse_pos = (0, 0)
        self.world_width = WIDTH
        self.world_height = HEIGHT
        self.bev_window = None
        self.show_bev_live = False

        # ========== 扩展画布（顶部不扩展，左右和底部扩展） ==========
        # 收集所有点的坐标（源点 + 计数线 + 图像四角）
        all_x = [p[0] for p in self.src_points] + [p[0] for p in self.line_points] + [0, self.w]
        all_y = [p[1] for p in self.src_points] + [p[1] for p in self.line_points] + [0, self.h]

        margin_x = 200  # 左右额外空间
        margin_bottom = 200  # 底部扩展
        self.canvas_w = self.w +  margin_x
        self.canvas_h = self.h + margin_bottom
        # ✅ 水平居中
        self.offset_x = (self.canvas_w - self.w) // 2
        # ✅ 顶部对齐
        self.offset_y = 0

        self.canvas = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8) * 255
        # 将原图放在画布中央水平位置
        self.canvas[self.offset_y:self.offset_y + self.h, self.offset_x:self.offset_x + self.w] = self.frame
        self.display_canvas = self.canvas.copy()

        self.window_name = "Interactive Calibration (Extended)"

    def to_canvas_coords(self, x, y):
        return int(x + self.offset_x), int(y)  # y 无偏移

    def from_canvas_coords(self, cx, cy):
        return cx - self.offset_x, cy

    def run(self):
        """运行交互标定流程"""
        if len(self.src_points) == 0:
            margin = 50
            self.src_points = [(margin, margin), (self.w-margin, margin),
                               (self.w-margin, self.h-margin), (margin, self.h-margin)]
            self.update_default_world_points()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("🎯 交互式标定工具使用说明（扩展画布版）")
        print("=" * 60)
        print("【鼠标操作】")
        print("  • 左键拖动点 → 移动点位置")
        print("  • 右键点击点 → 编辑该点的世界坐标")
        print("  • 鼠标悬停 → 显示坐标信息")
        print("\n【键盘命令】")
        print("  • [1] 编辑透视点（红色）")
        print("  • [2] 编辑计数线（蓝色）")
        print("  • [3] 编辑车道线（绿色）- 点击画线，自动拟合为直线")
        print("  • [C] 编辑所有点的世界坐标")
        print("  • [R] 重置所有点")
        print("  • [U] 撤销最后一次点击")
        print("  • [P] 预览俯视图变换效果")
        print("  • [L] 实时BEV预览模式")
        print("  • [ENTER] 完成标定并保存")
        print("  • [ESC] 取消标定")
        print("=" * 60)

        while True:
            self.update_display()
            cv2.imshow(self.window_name, self.display_canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:
                if len(self.src_points) == 4 and len(self.line_points) == 2:
                    break
                else:
                    print(f"❌ 需要4个透视点（当前{len(self.src_points)}）和2个计数线点（当前{len(self.line_points)}）")
            elif key == 27:
                self.close_bev_preview()
                return None
            elif key == ord('1'):
                self.active_mode = "src"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑透视点（红色）")
            elif key == ord('2'):
                self.active_mode = "line"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑计数线（蓝色）")
            elif key == ord('3'):
                self.active_mode = "lane"
                self.lane_calib_mode = True
                print("✅ 当前模式：编辑车道线（绿色）")
            elif key == ord('u') or key == ord('U'):
                # 撤销逻辑（保持原有，略）
                if self.lane_calib_mode and hasattr(self, 'lane_temp_points') and self.lane_temp_points:
                    self.lane_temp_points.pop()
                    print(f"🗑️ 撤销一点，剩余 {len(self.lane_temp_points)} 个点")
                    if len(self.lane_temp_points) >= 2:
                        self.fit_lane_line()
                elif self.active_mode == "src" and self.src_points:
                    self.src_points.pop()
                    print(f"🗑️ 撤销透视点，剩余 {len(self.src_points)} 个点")
                elif self.active_mode == "line" and self.line_points:
                    self.line_points.pop()
                    print(f"🗑️ 撤销计数线点，剩余 {len(self.line_points)} 个点")
            elif key == ord('c') or key == ord('C'):
                self.edit_all_world_coords()
            elif key == ord('r') or key == ord('R'):
                self.reset_points()
            elif key == ord('p') or key == ord('P'):
                self.preview_bird_eye_view()
            elif key == ord('l') or key == ord('L'):
                self.toggle_bev_live_preview()

        self.close_bev_preview()
        cv2.destroyWindow(self.window_name)

        src_array = np.array(self.src_points, dtype=np.float32)
        world_array = np.array(self.world_points, dtype=np.float32)
        return src_array, world_array, self.line_points[0], self.line_points[1], self.lane_lines

    def mouse_callback(self, event, canvas_x, canvas_y, flags, param):
        """鼠标回调 - 将画布坐标转为原始图像坐标"""
        x, y = self.from_canvas_coords(canvas_x, canvas_y)
        self.mouse_pos = (x, y)

        if self.lane_calib_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                if not hasattr(self, 'lane_temp_points'):
                    self.lane_temp_points = []
                self.lane_temp_points.append((x, y))
                print(f"📍 车道标定点 {len(self.lane_temp_points)}: ({x}, {y})")
                if len(self.lane_temp_points) >= 2:
                    self.fit_lane_line()
                self.update_display()
            return

        points = self.src_points if self.active_mode == "src" else self.line_points

        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self.get_hover_point(x, y, points)

        elif event == cv2.EVENT_LBUTTONDOWN:
            idx = self.get_hover_point(x, y, points)
            if idx >= 0:
                self.dragging_idx = idx
            else:
                if self.active_mode == "src" and len(points) < 4:
                    points.append((x, y))
                    if len(points) == 4:
                        self.update_default_world_points()
                    print(f"✅ 添加透视点 {len(points)}/4: ({x}, {y})")
                elif self.active_mode == "line" and len(points) < 2:
                    points.append((x, y))
                    print(f"✅ 添加计数线点 {len(points)}/2: ({x}, {y})")

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self.get_hover_point(x, y, self.src_points)
            if idx >= 0 and self.active_mode == "src":
                self.edit_point_world_coord(idx)

    def get_hover_point(self, x, y, points):
        for i, (px, py) in enumerate(points):
            if abs(x - px) < 8 and abs(y - py) < 8:
                return i
        return -1

    def update_display(self):
        """更新显示画面（文字提示限制在图像区域内）"""
        # 重置画布为灰色背景
        self.display_canvas = np.full((self.canvas_h, self.canvas_w, 3), 128, dtype=np.uint8)
        # 放置原始图像
        self.display_canvas[self.offset_y:self.offset_y + self.h,
        self.offset_x:self.offset_x + self.w] = self.frame

        # 根据当前模式绘制不同颜色的点
        if self.active_mode == "src":
            self.draw_points(self.src_points, (0, 0, 255), "src")
            self.draw_points(self.line_points, (255, 0, 0), "line", alpha=0.5)
        elif self.active_mode == "line":
            self.draw_points(self.src_points, (0, 0, 255), "src", alpha=0.5)
            self.draw_points(self.line_points, (255, 0, 0), "line")
        else:  # lane mode
            self.draw_points(self.src_points, (0, 0, 255), "src", alpha=0.5)
            self.draw_points(self.line_points, (255, 0, 0), "line", alpha=0.5)


        # ========== 车道线可视化 ==========
        if len(self.lane_lines) >= 2:
            overlay = self.display_canvas.copy()
            for i in range(len(self.lane_lines) - 1):
                x1a, y1a, x2a, y2a = self.lane_lines[i]
                x1b, y1b, x2b, y2b = self.lane_lines[i + 1]
                pts = np.array([[x1a, y1a], [x2a, y2a], [x2b, y2b], [x1b, y1b]], np.int32)
                pts_canvas = np.array([self.to_canvas_coords(p[0], p[1]) for p in pts.reshape(-1, 2)]).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts_canvas], (100, 0, 255))
            cv2.addWeighted(overlay, 0.2, self.display_canvas, 0.8, 0, self.display_canvas)
        for i, (x1, y1, x2, y2) in enumerate(self.lane_lines) :
            cx1, cy1 = self.to_canvas_coords(x1, y1)
            cx2, cy2 = self.to_canvas_coords(x2, y2)
            length = int(np.hypot(cx2 - cx1, cy2 - cy1))
            dash_len, solid_len = 20, 25
            if length > dash_len + solid_len:
                dx = (cx2 - cx1) / length
                dy = (cy2 - cy1) / length
                current = 0
                while current < length:
                    start_pt = (int(cx1 + dx * current), int(cy1 + dy * current))
                    end_pt = (int(cx1 + dx * min(current + dash_len, length)),
                              int(cy1 + dy * min(current + dash_len, length)))
                    cv2.line(self.display_canvas, start_pt, end_pt, (0, 255, 255), 3, cv2.LINE_AA)
                    current += dash_len + solid_len
            else:
                cv2.line(self.display_canvas, (cx1, cy1), (cx2, cy2), (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(self.display_canvas, (cx1, cy1), 5, (0, 255, 255), -1)
            cv2.circle(self.display_canvas, (cx2, cy2), 5, (0, 255, 255), -1)

            # 车道线编号（在画布上绘制，但位置在原始图像区域内？通常车道线本身在图像内，无需额外限制）
            mid_x, mid_y = (cx1 + cx2) // 2, (cy1 + cy2) // 2
            text = f"Lane {i + 1}"
            font_scale, thickness = 0.7, 2
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            box_x, box_y = mid_x - tw // 2 - 5, mid_y - th // 2 - 5
            cv2.rectangle(self.display_canvas, (box_x, box_y), (box_x + tw + 10, box_y + th + 10), (0, 0, 0), -1)
            cv2.putText(self.display_canvas, text, (mid_x - tw // 2, mid_y + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        # 临时车道标定点
        if hasattr(self, 'lane_temp_points') and self.lane_temp_points:
            for (x, y) in self.lane_temp_points:
                cx, cy = self.to_canvas_coords(x, y)
                cv2.circle(self.display_canvas, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(self.display_canvas, (cx, cy), 7, (0, 255, 255), 2)
            if len(self.lane_temp_points) >= 2:
                points = np.array(self.lane_temp_points)
                x_std = np.std(points[:, 0])
                if x_std < 10:
                    x_center = int(np.mean(points[:, 0]))
                    cx1, cy1 = self.to_canvas_coords(x_center, 0)
                    cx2, cy2 = self.to_canvas_coords(x_center, self.h)
                    cv2.line(self.display_canvas, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
                else:
                    x = points[:, 0].reshape(-1, 1)
                    y = points[:, 1].reshape(-1, 1)
                    A = np.hstack([y, np.ones_like(y)])
                    params = np.linalg.lstsq(A, x, rcond=None)[0]
                    a, b = params[0][0], params[1][0]
                    x1 = int(a * 0 + b)
                    x2 = int(a * self.h + b)
                    cx1, cy1 = self.to_canvas_coords(x1, 0)
                    cx2, cy2 = self.to_canvas_coords(x2, self.h)
                    cv2.line(self.display_canvas, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)


        # 计数线
        if len(self.line_points) == 2:
            x1, y1 = self.line_points[0]
            x2, y2 = self.line_points[1]
            cx1, cy1 = self.to_canvas_coords(x1, y1)
            cx2, cy2 = self.to_canvas_coords(x2, y2)
            cv2.line(self.display_canvas, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
            self.draw_arrow((cx1, cy1), (cx2, cy2), (255, 0, 0))

        # 透视四边形
        if len(self.src_points) == 4:
            pts = np.array(self.src_points, np.int32)
            pts_canvas = np.array([self.to_canvas_coords(p[0], p[1]) for p in pts])
            cv2.polylines(self.display_canvas, [pts_canvas], True, (0, 255, 255), self.line_thickness)
            for i in range(4):
                j = (i + 1) % 4
                start = pts_canvas[i]
                end = pts_canvas[j]
                self.draw_arrow(tuple(start), tuple(end), (0, 255, 0), 1, 0.2)



        # ========== 文字提示（限制在图像区域内） ==========
        img_left = self.offset_x
        img_top = self.offset_y
        img_right = self.offset_x + self.w
        img_bottom = self.offset_y + self.h

        # 1. 模式文字（图像左上角）
        if self.lane_calib_mode:
            mode_text = f"Lane Mode ({len(self.lane_lines)} lanes, {len(self.lane_temp_points) if hasattr(self, 'lane_temp_points') else 0} temp points)"
        else:
            mode_text = "Perspective Mode" if self.active_mode == "src" else "Count Line Mode"
        text_pos = (img_left + 10, img_top + 30)
        cv2.putText(self.display_canvas, mode_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 2. 鼠标坐标（图像右上角）
        cursor_text = f"Cursor: ({self.mouse_pos[0]}, {self.mouse_pos[1]})"
        (tw, th), _ = cv2.getTextSize(cursor_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cursor_x = img_right - tw - 10
        cursor_y = img_top + 30
        cv2.putText(self.display_canvas, cursor_text, (cursor_x, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        # 3. BEV 状态（紧挨着鼠标坐标左侧）
        bev_status = "BEV Live: ON" if self.show_bev_live else "BEV Live: OFF"
        (bw, bh), _ = cv2.getTextSize(bev_status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bev_x = cursor_x - bw - 10
        if bev_x > img_left + 10:
            color = (0, 255, 0) if self.show_bev_live else (0, 0, 255)
            cv2.putText(self.display_canvas, bev_status, (bev_x, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 4. 鼠标悬停信息（限制在图像区域内）
        if self.hover_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode == "src" else self.line_points
            if self.hover_idx < len(points):
                pt = points[self.hover_idx]
                if self.active_mode == "src" and self.hover_idx < len(self.world_points):
                    world = self.world_points[self.hover_idx]
                    coord_text = f"P{self.hover_idx + 1}: img({pt[0]},{pt[1]}) world({world[0]:.1f},{world[1]:.1f})m"
                else:
                    coord_text = f"L{self.hover_idx + 1}: ({pt[0]},{pt[1]})"
                # 鼠标在画布上的位置（用于定位提示框）
                mouse_cx, mouse_cy = self.to_canvas_coords(self.mouse_pos[0], self.mouse_pos[1])
                # 限制文本位置在图像区域内
                text_cx = max(img_left + 10, min(img_right - 200, mouse_cx + 10))
                text_cy = max(img_top + 20, min(img_bottom - 20, mouse_cy - 10))
                (tw2, th2), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(self.display_canvas,
                              (text_cx - 2, text_cy - th2 - 2),
                              (text_cx + tw2 + 2, text_cy + 2),
                              (0, 0, 0), -1)
                cv2.putText(self.display_canvas, coord_text, (text_cx, text_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 拖动更新点位置（保持原有逻辑）
        if self.dragging_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode == "src" else self.line_points
            if self.dragging_idx < len(points):
                x, y = self.mouse_pos
                # 限制坐标范围（允许为负，但为了不超出画布，可以设置边界为 [-offset_x, canvas_w-offset_x]）
                x = max(-self.offset_x, min(self.canvas_w - self.offset_x, x))
                y = max(-self.offset_y, min(self.canvas_h - self.offset_y, y))
                points[self.dragging_idx] = (x, y)
                if self.active_mode == "src" and len(self.src_points) == 4:
                    pass  # 不自动更新世界坐标
                if self.show_bev_live and self.active_mode == "src" and len(self.src_points) == 4:
                    self.update_bev_live_preview()

    def draw_points(self, points, color, point_type, alpha=1.0):
        """绘制点（坐标转换为画布坐标）"""
        for i, (x,y) in enumerate(points):
            cx, cy = self.to_canvas_coords(x, y)
            if 0 <= cx < self.canvas_w and 0 <= cy < self.canvas_h:
                cv2.circle(self.display_canvas, (cx, cy), 6, color, -1)
                cv2.circle(self.display_canvas, (cx, cy), 8, color, 2)
                cv2.putText(self.display_canvas, str(i+1), (cx+10, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.hover_idx == i and not self.lane_calib_mode and \
                    ((self.active_mode=="src" and point_type=="src") or (self.active_mode=="line" and point_type=="line")):
                if 0 <= cx < self.canvas_w and 0 <= cy < self.canvas_h:
                    cv2.circle(self.display_canvas, (cx, cy), 12, (255,255,0), 2)

    def draw_arrow(self, start, end, color, thickness=2, tip_length=0.3):
        # 与之前相同，但输入的 start/end 已为画布坐标
        start_pt = np.array(start)
        end_pt = np.array(end)
        direction = end_pt - start_pt
        length = np.linalg.norm(direction)
        if length < 1:
            return
        direction = direction / length
        perp = np.array([-direction[1], direction[0]])
        arrow_size = min(20, length * tip_length)
        tip = end_pt - direction * arrow_size
        left = tip + perp * (arrow_size * 0.5)
        right = tip - perp * (arrow_size * 0.5)
        pts = np.array([end_pt, left, tip, right], dtype=np.int32)
        cv2.fillPoly(self.display_canvas, [pts], color)

    def fit_lane_line(self):
        # 与原逻辑相同，使用原始图像坐标
        if not hasattr(self, 'lane_temp_points') or len(self.lane_temp_points) < 2:
            return
        points = np.array(self.lane_temp_points)
        y_min = np.min(points[:,1])
        y_max = np.max(points[:,1])
        x_std = np.std(points[:,0])
        if x_std < 10:
            x_center = int(np.mean(points[:,0]))
            lane_line = (x_center, int(y_min), x_center, int(y_max))
        else:
            x = points[:,0].reshape(-1,1)
            y = points[:,1].reshape(-1,1)
            A = np.hstack([y, np.ones_like(y)])
            params = np.linalg.lstsq(A, x, rcond=None)[0]
            a,b = params[0][0], params[1][0]
            x1 = int(a * y_min + b)
            x2 = int(a * y_max + b)
            lane_line = (x1, int(y_min), x2, int(y_max))
        self.lane_lines.append(lane_line)
        print(f"✅ 已添加车道线 {len(self.lane_lines)}: ({lane_line[0]},{lane_line[1]})->({lane_line[2]},{lane_line[3]})")
        self.lane_temp_points = []
        self.update_display()

    def edit_all_world_coords(self):
        # 保持不变（使用世界坐标列表）
        import tkinter as tk
        from tkinter import ttk
        if len(self.world_points) != 4:
            print("❌ 需要先有4个透视点")
            return
        root = tk.Tk()
        root.title("编辑所有点的世界坐标 (X, Y 米)")
        root.geometry("400x300")
        root.attributes('-topmost', True)
        entries = []
        ttk.Label(root, text="点序号", font=("Arial",10,"bold")).grid(row=0,column=0,padx=5,pady=5)
        ttk.Label(root, text="X (米)", font=("Arial",10,"bold")).grid(row=0,column=1,padx=5,pady=5)
        ttk.Label(root, text="Y (米)", font=("Arial",10,"bold")).grid(row=0,column=2,padx=5,pady=5)
        for i, (wx, wy) in enumerate(self.world_points):
            ttk.Label(root, text=f"点 {i+1}").grid(row=i+1, column=0, padx=5, pady=5)
            var_x = tk.StringVar(value=f"{wx:.2f}")
            var_y = tk.StringVar(value=f"{wy:.2f}")
            tk.Entry(root, textvariable=var_x, width=10).grid(row=i+1, column=1, padx=5, pady=5)
            tk.Entry(root, textvariable=var_y, width=10).grid(row=i+1, column=2, padx=5, pady=5)
            entries.append((var_x, var_y))
        def on_confirm():
            try:
                new_pts = [(float(vx.get()), float(vy.get())) for vx,vy in entries]
                self.world_points = new_pts
                xs = [p[0] for p in self.world_points]
                ys = [p[1] for p in self.world_points]
                self.world_width = max(xs)-min(xs)
                self.world_height = max(ys)-min(ys)
                print(f"✅ 世界坐标已更新: {self.world_points}")
                root.destroy()
                self.update_display()
                if self.show_bev_live:
                    self.update_bev_live_preview()
            except:
                tk.messagebox.showerror("错误","请输入有效数字")
        def on_cancel():
            root.destroy()
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=15)
        tk.Button(btn_frame, text="确定", command=on_confirm).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="取消", command=on_cancel).pack(side=tk.LEFT, padx=10)
        root.mainloop()

    def edit_point_world_coord(self, idx):
        # 与原逻辑相同，控制台输入
        if idx >= len(self.world_points):
            return
        current = self.world_points[idx]
        print(f"\n编辑点 {idx+1} 的世界坐标（当前: x={current[0]:.2f}m, y={current[1]:.2f}m）")
        try:
            x_in = input("新的X坐标（米）: ").strip()
            y_in = input("新的Y坐标（米）: ").strip()
            new_x = float(x_in) if x_in else current[0]
            new_y = float(y_in) if y_in else current[1]
            self.world_points[idx] = (new_x, new_y)
            print(f"✅ 已更新")
            if len(self.world_points)==4:
                xs = [p[0] for p in self.world_points]
                ys = [p[1] for p in self.world_points]
                self.world_width = max(xs)-min(xs)
                self.world_height = max(ys)-min(ys)
            if self.show_bev_live:
                self.update_bev_live_preview()
        except:
            print("❌ 输入无效")

    def reset_points(self):
        margin = 50
        self.src_points = [(margin, margin), (self.w-margin, margin),
                           (self.w-margin, self.h-margin), (margin, self.h-margin)]
        self.update_default_world_points()
        self.line_points = [(OFFSET, LINE_Y), (self.w-OFFSET, LINE_Y)]
        self.lane_lines = []
        if hasattr(self, 'lane_temp_points'):
            self.lane_temp_points = []
        print("✅ 已重置所有点")
        if self.show_bev_live:
            self.update_bev_live_preview()

    def update_default_world_points(self):
        if len(self.src_points) == 4:
            self.world_points = [(0,0), (self.world_width,0), (self.world_width,self.world_height), (0,self.world_height)]

    def preview_bird_eye_view(self):
        # 与原有相同（使用原始图像和坐标）
        if len(self.src_points) != 4:
            print("❌ 需要4个透视点")
            return
        try:
            src = np.array(self.src_points, dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)
            if self.polygon_area(src) * self.polygon_area(dst) < 0:
                dst = dst[::-1]
            M = cv2.getPerspectiveTransform(src, dst)
            world_width = max(dst[:,0])-min(dst[:,0])
            world_height = max(dst[:,1])-min(dst[:,1])
            target_size = 800
            scale = target_size / max(world_width, world_height) if world_width>0 else 20
            scale = max(10, min(100, scale))
            w_dst = max(100, int(world_width*scale))
            h_dst = max(100, int(world_height*scale))
            bev = cv2.warpPerspective(self.frame, M, (w_dst, h_dst))
            bev = self.add_reference_grid(bev, world_width, world_height, scale)
            cv2.imshow("BEV Preview", bev)
            cv2.waitKey(0)
            cv2.destroyWindow("BEV Preview")
        except Exception as e:
            print(f"❌ 预览失败: {e}")

    def toggle_bev_live_preview(self):
        if not self.show_bev_live:
            if len(self.src_points)==4:
                self.show_bev_live=True
                self.create_bev_preview_window()
                print("✅ 实时BEV预览已开启")
            else:
                print("❌ 需要4个透视点")
        else:
            self.close_bev_preview()
            print("✅ 实时BEV预览已关闭")

    def create_bev_preview_window(self):
        self.bev_window = "BEV Live Preview"
        cv2.namedWindow(self.bev_window)

        # 更新BEV
        self.update_bev_live_preview()

    def update_bev_live_preview(self):
        if not self.show_bev_live or self.bev_window is None:
            return
        if len(self.src_points)!=4:
            return
        try:
            src = np.array(self.src_points[:4], dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)
            # 保证方向一致
            if self.polygon_area(src) * self.polygon_area(dst) < 0:
                dst = dst[::-1]
            # ---------- 计算真实尺度 ----------
            world_width = max(dst[:, 0]) - min(dst[:, 0])
            world_height = max(dst[:, 1]) - min(dst[:, 1])
            PIX_PER_METER = 10  # ⭐建议固定，不要动态scale
            bev_w = int(world_width * PIX_PER_METER)
            bev_h = int(world_height * PIX_PER_METER)
            bev_w = max(200, bev_w)
            bev_h = max(200, bev_h)
            # ---------- 目标像素坐标 ----------
            dst_pixels = np.array([
                [0, 0],
                [bev_w, 0],
                [bev_w, bev_h],
                [0, bev_h]
            ], dtype=np.float32)
            # ---------- 透视变换 ----------
            M = cv2.getPerspectiveTransform(src, dst_pixels)
            bev = cv2.warpPerspective(
                self.frame,
                M,
                (bev_w, bev_h)
            )
            # ---------- 加网格 ----------
            bev = self.add_reference_grid(
                bev,
                world_width,
                world_height,
                PIX_PER_METER
            )

            cv2.imshow(self.bev_window, bev)
            cv2.waitKey(1)
        except:
            pass

    def close_bev_preview(self):
        if self.bev_window is not None:
            self.show_bev_live = False
            try:
                cv2.destroyWindow(self.bev_window)
            except:
                pass
            self.bev_window = None

    def polygon_area(self, points):
        area = 0
        n = len(points)
        for i in range(n):
            x1,y1 = points[i]
            x2,y2 = points[(i+1)%n]
            area += x1*y2 - x2*y1
        return area/2

    def add_reference_grid(self, img, world_width, world_height, scale):
        h,w = img.shape[:2]
        grid_spacing = max(1, int(min(world_width, world_height)/5))
        grid_px = max(10, int(grid_spacing*scale))
        for x in range(0, w, grid_px):
            cv2.line(img, (x,0), (x,h), (0,255,0), 1)
            if x+5<w:
                cv2.putText(img, f"{int(x/scale)}m", (x+2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
        for y in range(0, h, grid_px):
            cv2.line(img, (0,y), (w,y), (0,255,0), 1)
            if y+10<h:
                cv2.putText(img, f"{int(y/scale)}m", (2,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
        return img
class InteractiveCalibrationExtendCav:
    """高交互式标定工具 - 支持拖动点、编辑坐标、实时预览（扩展画布版）"""
    def __init__(self, frame):
        """
        参数:
            frame: 原始图像帧（numpy array）
        """
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.line_thickness = 1  # 统一线条粗细

        # ----- 标定数据（与原始坐标一致）-----
        self.src_points = SOURCE_POINTS          # 图像上的4个透视点 [(x,y), ...]
        self.world_points = TARGET_POINTS        # 对应的世界坐标点 [(x_m, y_m), ...]
        self.line_points = [(OFFSET, LINE_Y), (self.w - OFFSET, LINE_Y)]

        # 车道线相关
        self.lane_lines = []
        self.lane_calib_mode = False
        self.lane_temp_points = []               # 临时点

        # 交互状态
        self.dragging_idx = -1
        self.active_mode = "src"                 # "src", "line", "lane"
        self.hover_idx = -1
        self.mouse_pos = (0, 0)

        # 世界范围（用于显示）
        self.world_width = WIDTH
        self.world_height = HEIGHT

        # BEV预览
        self.bev_window = None
        self.show_bev_live = False

        # ========== 扩展画布参数（水平居中，顶部对齐，底部扩展） ==========
        margin_x = 200  # 左右额外空间
        margin_bottom = 200  # 底部扩展
        self.canvas_w = self.w + margin_x
        self.canvas_h = self.h + margin_bottom
        # ✅ 水平居中
        self.offset_x = (self.canvas_w - self.w) // 2
        # ✅ 顶部对齐
        self.offset_y = 0

        # 创建白色背景画布
        self.canvas = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        # 将原图放在画布水平居中、顶部对齐的位置
        self.canvas[self.offset_y:self.offset_y+self.h,
                    self.offset_x:self.offset_x+self.w] = self.frame

        self.window_name = "Interactive Calibration (Extended Canvas)"
    # ------------------- 坐标转换辅助函数 -------------------
    def to_canvas_coords(self, x, y):
        """原始图像坐标 -> 画布坐标"""
        return int(x + self.offset_x), int(y + self.offset_y)

    def from_canvas_coords(self, cx, cy):
        """画布坐标 -> 原始图像坐标"""
        return cx - self.offset_x, cy - self.offset_y

    # ------------------- 主循环 -------------------
    def run(self):
        """运行交互标定流程，返回 (SOURCE, TARGET, line_start, line_end, lane_lines)"""
        # 若没有透视点，初始化默认点（图像四角偏移 margin 像素）
        if len(self.src_points) == 0:
            m = 50
            self.src_points = [(m, m), (self.w - m, m),
                               (self.w - m, self.h - m), (m, self.h - m)]
            self.update_default_world_points()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self._print_help()

        while True:
            self.update_display()
            cv2.imshow(self.window_name, self.canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:          # Enter
                if len(self.src_points) == 4 and len(self.line_points) == 2:
                    break
                else:
                    print(f"❌ 需要4个透视点（当前{len(self.src_points)}）"
                          f"和2个计数线点（当前{len(self.line_points)}）")
            elif key == 27:        # ESC
                self.close_bev_preview()
                return None
            elif key == ord('1'):
                self.active_mode = "src"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑透视点（红色）")
            elif key == ord('2'):
                self.active_mode = "line"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑计数线（蓝色）")
            elif key == ord('3'):
                self.active_mode = "lane"
                self.lane_calib_mode = True
                print("✅ 当前模式：编辑车道线（绿色）")
                print("   提示：在图像上点击多个点，系统会自动拟合为直线")
            elif key == ord('u') or key == ord('U'):
                self._undo()
            elif key == ord('w') or key == ord('W'):
                self.set_world_range()
            elif key == ord('r') or key == ord('R'):
                self.reset_points()
            elif key == ord('p') or key == ord('P'):
                self.preview_bird_eye_view()
            elif key == ord('l') or key == ord('L'):
                self.toggle_bev_live_preview()
            elif key == ord('s') or key == ord('S'):
                self.set_road_preset()

        self.close_bev_preview()
        cv2.destroyWindow(self.window_name)

        # 返回标定数据（保持原始坐标）
        src_array = np.array(self.src_points, dtype=np.float32)
        world_array = np.array(self.world_points, dtype=np.float32)
        return src_array, world_array, self.line_points[0], self.line_points[1], self.lane_lines

    def _print_help(self):
        print("\n" + "=" * 60)
        print("🎯 交互式标定工具使用说明（扩展画布版）")
        print("=" * 60)
        print("【鼠标操作】")
        print("  • 左键拖动点 → 移动点位置")
        print("  • 右键点击点 → 编辑该点的世界坐标")
        print("  • 鼠标悬停 → 显示坐标信息")
        print("\n【键盘命令】")
        print("  • [1] 编辑透视点（红色）")
        print("  • [2] 编辑计数线（蓝色）")
        print("  • [3] 编辑车道线（绿色）- 点击画线，自动拟合为直线")
        print("  • [W] 设置世界范围（宽/高）")
        print("  • [R] 重置所有点")
        print("  • [U] 撤销最后一次点击")
        print("  • [P] 预览俯视图变换效果")
        print("  • [L] 实时BEV预览模式")
        print("  • [S] 道路标准预设")
        print("  • [ENTER] 完成标定并保存")
        print("  • [ESC] 取消标定")
        print("=" * 60)

    def _undo(self):
        """撤销操作"""
        if self.lane_calib_mode and self.lane_temp_points:
            self.lane_temp_points.pop()
            print(f"🗑️ 撤销一点，剩余 {len(self.lane_temp_points)} 个点")
            if len(self.lane_temp_points) >= 2:
                self.fit_lane_line()
        elif self.active_mode == "src" and self.src_points:
            self.src_points.pop()
            print(f"🗑️ 撤销透视点，剩余 {len(self.src_points)} 个点")
        elif self.active_mode == "line" and self.line_points:
            self.line_points.pop()
            print(f"🗑️ 撤销计数线点，剩余 {len(self.line_points)} 个点")

    # ------------------- 鼠标回调（坐标转换） -------------------
    def mouse_callback(self, event, canvas_x, canvas_y, flags, param):
        # 将画布坐标转换为原始图像坐标
        x, y = self.from_canvas_coords(canvas_x, canvas_y)
        self.mouse_pos = (x, y)

        if self.lane_calib_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.lane_temp_points.append((x, y))
                print(f"📍 车道标定点 {len(self.lane_temp_points)}: ({x}, {y})")
                if len(self.lane_temp_points) >= 2:
                    self.fit_lane_line()
                self.update_display()
            return

        points = self.src_points if self.active_mode == "src" else self.line_points

        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self.get_hover_point(x, y, points)

        elif event == cv2.EVENT_LBUTTONDOWN:
            idx = self.get_hover_point(x, y, points)
            if idx >= 0:
                self.dragging_idx = idx
            else:
                if self.active_mode == "src" and len(points) < 4:
                    points.append((x, y))
                    if len(points) == 4:
                        self.update_default_world_points()
                    print(f"✅ 添加透视点 {len(points)}/4: ({x}, {y})")
                elif self.active_mode == "line" and len(points) < 2:
                    points.append((x, y))
                    print(f"✅ 添加计数线点 {len(points)}/2: ({x}, {y})")

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self.get_hover_point(x, y, self.src_points)
            if idx >= 0 and self.active_mode == "src":
                self.edit_point_world_coord(idx)

    # ------------------- 几何辅助函数 -------------------
    def get_hover_point(self, x, y, points):
        for i, (px, py) in enumerate(points):
            if abs(x - px) < 8 and abs(y - py) < 8:
                return i
        return -1

    def fit_lane_line(self):
        """将临时点拟合为直线（保留原始坐标）"""
        if len(self.lane_temp_points) < 2:
            return
        pts = np.array(self.lane_temp_points)
        y_min = int(np.min(pts[:, 1]))
        y_max = int(np.max(pts[:, 1]))
        x_std = np.std(pts[:, 0])
        if x_std < 10:
            x_center = int(np.mean(pts[:, 0]))
            lane_line = (x_center, y_min, x_center, y_max)
        else:
            x = pts[:, 0].reshape(-1, 1)
            y = pts[:, 1].reshape(-1, 1)
            A = np.hstack([y, np.ones_like(y)])
            params = np.linalg.lstsq(A, x, rcond=None)[0]
            a, b = params[0][0], params[1][0]
            x1 = int(a * y_min + b)
            x2 = int(a * y_max + b)
            lane_line = (x1, y_min, x2, y_max)
        self.lane_lines.append(lane_line)
        print(f"✅ 已添加车道线 {len(self.lane_lines)}: {lane_line}")
        self.lane_temp_points.clear()
        self.update_display()

    # ------------------- 显示更新（所有绘图均转换到画布坐标） -------------------
    def update_display(self):
        """更新显示（画布）"""
        # 重置画布为白色并放置原图
        self.canvas = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8) * 255
        self.canvas[self.offset_y:self.offset_y+self.h,
                    self.offset_x:self.offset_x+self.w] = self.frame

        # 绘制点和线（使用转换后的坐标）
        self.draw_points(self.src_points, (0, 0, 255), "src")
        self.draw_points(self.line_points, (255, 0, 0), "line", alpha=0.5 if self.active_mode != "line" else 1.0)

        # 车道线可视化
        self._draw_lanes()
        self._draw_temp_lane_points()

        # 计数线
        if len(self.line_points) == 2:
            p1 = self.to_canvas_coords(*self.line_points[0])
            p2 = self.to_canvas_coords(*self.line_points[1])
            cv2.line(self.canvas, p1, p2, (255, 0, 0), 2)
            self.draw_arrow(p1, p2, (255, 0, 0))

        # 透视四边形
        if len(self.src_points) == 4:
            pts_canvas = [self.to_canvas_coords(px, py) for px, py in self.src_points]
            cv2.polylines(self.canvas, [np.array(pts_canvas, np.int32)], True, (0, 255, 255), self.line_thickness)
            for i in range(4):
                j = (i + 1) % 4
                self.draw_arrow(pts_canvas[i], pts_canvas[j], (0, 255, 0), 1, 0.2)

        # 文字提示（使用原始图像区域内的画布坐标）
        self._draw_text()

        # 拖动时实时更新点位置
        if self.dragging_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode == "src" else self.line_points
            if self.dragging_idx < len(points):
                x, y = self.mouse_pos
                # 限制坐标范围（不允许超出画布对应原始图像范围，但允许负值）
                x = max(-self.offset_x, min(self.canvas_w - self.offset_x, x))
                y = max(-self.offset_y, min(self.canvas_h - self.offset_y, y))
                points[self.dragging_idx] = (x, y)
                if self.active_mode == "src" and len(self.src_points) == 4:
                    # 拖动时不再自动更新世界坐标（保留用户手动设定）
                    pass
                if self.show_bev_live and self.active_mode == "src" and len(self.src_points) == 4:
                    self.update_bev_live_preview()

    # ------------------- 绘图子函数（坐标转换） -------------------
    def draw_points(self, points, color, point_type, alpha=1.0):
        for i, (x, y) in enumerate(points):
            cx, cy = self.to_canvas_coords(x, y)
            # 只绘制在画布内的点（超出部分忽略，不绘制）
            if 0 <= cx < self.canvas_w and 0 <= cy < self.canvas_h:
                cv2.circle(self.canvas, (cx, cy), 6, color, -1)
                cv2.circle(self.canvas, (cx, cy), 8, color, 2)
                cv2.putText(self.canvas, str(i+1), (cx+10, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # 悬停高亮
                if self.hover_idx == i and not self.lane_calib_mode and \
                        ((self.active_mode == "src" and point_type == "src") or
                         (self.active_mode == "line" and point_type == "line")):
                    cv2.circle(self.canvas, (cx, cy), 12, (255, 255, 0), 2)

    def draw_arrow(self, start, end, color, thickness=2, tip_length=0.3):
        start_pt = np.array(start)
        end_pt = np.array(end)
        direction = end_pt - start_pt
        length = np.linalg.norm(direction)
        if length < 1:
            return
        direction = direction / length
        perp = np.array([-direction[1], direction[0]])
        arrow_size = min(20, length * tip_length)
        tip = end_pt - direction * arrow_size
        left = tip + perp * (arrow_size * 0.5)
        right = tip - perp * (arrow_size * 0.5)
        pts = np.array([end_pt, left, tip, right], dtype=np.int32)
        cv2.fillPoly(self.canvas, [pts], color)

    def _draw_lanes(self):
        """绘制车道线（虚线 + 编号）"""
        if len(self.lane_lines) >= 2:
            overlay = self.canvas.copy()
            for i in range(len(self.lane_lines)-1):
                x1a,y1a,x2a,y2a = self.lane_lines[i]
                x1b,y1b,x2b,y2b = self.lane_lines[i+1]
                pts = np.array([[x1a,y1a],[x2a,y2a],[x2b,y2b],[x1b,y1b]], np.int32)
                pts_canvas = [self.to_canvas_coords(p[0], p[1]) for p in pts.reshape(-1,2)]
                cv2.fillPoly(overlay, [np.array(pts_canvas, np.int32)], (100,0,255))
            cv2.addWeighted(overlay, 0.2, self.canvas, 0.8, 0, self.canvas)

        for (x1,y1,x2,y2) in self.lane_lines:
            cx1,cy1 = self.to_canvas_coords(x1,y1)
            cx2,cy2 = self.to_canvas_coords(x2,y2)
            length = int(np.hypot(cx2-cx1, cy2-cy1))
            dash, solid = 20, 25
            if length > dash+solid:
                dx = (cx2-cx1)/length
                dy = (cy2-cy1)/length
                cur = 0
                while cur < length:
                    s = (int(cx1+dx*cur), int(cy1+dy*cur))
                    e = (int(cx1+dx*min(cur+dash, length)), int(cy1+dy*min(cur+dash, length)))
                    cv2.line(self.canvas, s, e, (0,255,255), 3, cv2.LINE_AA)
                    cur += dash+solid
            else:
                cv2.line(self.canvas, (cx1,cy1), (cx2,cy2), (0,255,255), 3, cv2.LINE_AA)

            # 车道编号
            mx, my = (cx1+cx2)//2, (cy1+cy2)//2
            text = f"Lane {self.lane_lines.index((x1,y1,x2,y2))+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
            box = (mx - tw//2 - 5, my - th//2 - 5)
            cv2.rectangle(self.canvas, box, (box[0]+tw+10, box[1]+th+10), (0,0,0), -1)
            cv2.putText(self.canvas, text, (mx - tw//2, my + th//2), font, scale, (0,255,255), 2)

    def _draw_temp_lane_points(self):
        """临时车道线点及预览"""
        if not self.lane_temp_points:
            return
        for (x,y) in self.lane_temp_points:
            cx,cy = self.to_canvas_coords(x,y)
            cv2.circle(self.canvas, (cx,cy), 5, (0,255,255), -1)
            cv2.circle(self.canvas, (cx,cy), 7, (0,255,255), 2)
        if len(self.lane_temp_points) >= 2:
            pts = np.array(self.lane_temp_points)
            x_std = np.std(pts[:,0])
            if x_std < 10:
                xc = int(np.mean(pts[:,0]))
                cx1,cy1 = self.to_canvas_coords(xc,0)
                cx2,cy2 = self.to_canvas_coords(xc,self.h)
                cv2.line(self.canvas, (cx1,cy1), (cx2,cy2), (0,255,255), 2)
            else:
                x = pts[:,0].reshape(-1,1)
                y = pts[:,1].reshape(-1,1)
                A = np.hstack([y, np.ones_like(y)])
                params = np.linalg.lstsq(A, x, rcond=None)[0]
                a,b = params[0][0], params[1][0]
                x1, x2 = int(a*0+b), int(a*self.h+b)
                cx1,cy1 = self.to_canvas_coords(x1,0)
                cx2,cy2 = self.to_canvas_coords(x2,self.h)
                cv2.line(self.canvas, (cx1,cy1), (cx2,cy2), (0,255,255), 2)

    def _draw_text(self):
        """绘制文字提示（限制在原始图像区域内）"""
        img_left = self.offset_x
        img_top = self.offset_y
        img_right = self.offset_x + self.w
        img_bottom = self.offset_y + self.h

        # 模式文字（左上角）
        if self.lane_calib_mode:
            mode_text = f"Lane Mode ({len(self.lane_lines)} lanes, {len(self.lane_temp_points)} temp)"
        else:
            mode_text = "Perspective Mode" if self.active_mode=="src" else "Count Line Mode"
        cv2.putText(self.canvas, mode_text, (img_left+10, img_top+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # 鼠标坐标（右上角）
        cursor_text = f"Cursor: ({self.mouse_pos[0]}, {self.mouse_pos[1]})"
        (tw, th), _ = cv2.getTextSize(cursor_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(self.canvas, cursor_text, (img_right-tw-10, img_top+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # BEV状态（紧挨鼠标坐标左侧）
        bev_status = "BEV Live: ON" if self.show_bev_live else "BEV Live: OFF"
        (bw, bh), _ = cv2.getTextSize(bev_status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bev_x = img_right - tw - bw - 20
        if bev_x > img_left+10:
            color = (0,255,0) if self.show_bev_live else (0,0,255)
            cv2.putText(self.canvas, bev_status, (bev_x, img_top+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 悬停信息（跟随鼠标，限制在图像区域内）
        if self.hover_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode=="src" else self.line_points
            if self.hover_idx < len(points):
                pt = points[self.hover_idx]
                if self.active_mode=="src" and self.hover_idx < len(self.world_points):
                    w = self.world_points[self.hover_idx]
                    coord_text = f"P{self.hover_idx+1}: img({pt[0]},{pt[1]}) world({w[0]:.1f},{w[1]:.1f})m"
                else:
                    coord_text = f"L{self.hover_idx+1}: ({pt[0]},{pt[1]})"
                # 计算鼠标在画布上的位置（用于定位提示框）
                mouse_cx, mouse_cy = self.to_canvas_coords(*self.mouse_pos)
                text_cx = max(img_left+10, min(img_right-200, mouse_cx+10))
                text_cy = max(img_top+20, min(img_bottom-20, mouse_cy-10))
                (tw2, th2), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(self.canvas,
                              (text_cx-2, text_cy-th2-2),
                              (text_cx+tw2+2, text_cy+2),
                              (0,0,0), -1)
                cv2.putText(self.canvas, coord_text, (text_cx, text_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # ------------------- 其他功能函数（保留原始坐标） -------------------
    def set_road_preset(self):
        self.world_width = 3.5
        self.world_height = 20.0
        self.update_default_world_points()
        print("✅ 已设置为道路标准参数（宽3.5m x 长20m）")
        if self.show_bev_live:
            self.update_bev_live_preview()

    def update_default_world_points(self):
        if len(self.src_points) == 4:
            self.world_points = [(0,0), (self.world_width,0),
                                 (self.world_width,self.world_height),
                                 (0,self.world_height)]

    def set_world_range(self):
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            new_w = simpledialog.askfloat("世界宽度", "请输入宽度（米）", initialvalue=self.world_width, parent=root)
            if new_w is not None:
                self.world_width = new_w
            new_h = simpledialog.askfloat("世界高度", "请输入长度（米）", initialvalue=self.world_height, parent=root)
            if new_h is not None:
                self.world_height = new_h
            root.destroy()
            self.update_default_world_points()
            print(f"✅ 已更新世界范围: 宽={self.world_width:.1f}m, 长={self.world_height:.1f}m")
        except Exception as e:
            print(f"UI对话框失败: {e}, 改用控制台")
            self._set_world_range_console()

    def _set_world_range_console(self):
        print(f"\n当前世界范围: 宽={self.world_width:.1f}m, 长={self.world_height:.1f}m")
        try:
            w_in = input(f"宽度（米）[默认{self.world_width:.1f}]: ").strip()
            if w_in:
                self.world_width = float(w_in)
            h_in = input(f"长度（米）[默认{self.world_height:.1f}]: ").strip()
            if h_in:
                self.world_height = float(h_in)
            self.update_default_world_points()
            print(f"✅ 已更新: {self.world_width:.1f}m x {self.world_height:.1f}m")
        except ValueError:
            print("❌ 输入无效")

    def edit_point_world_coord(self, idx):
        if idx >= len(self.world_points):
            return
        cur = self.world_points[idx]
        print(f"\n编辑点{idx+1}世界坐标（当前: X={cur[0]:.2f}m, Y={cur[1]:.2f}m）")
        try:
            x_in = input("新X（米）: ").strip()
            y_in = input("新Y（米）: ").strip()
            new_x = float(x_in) if x_in else cur[0]
            new_y = float(y_in) if y_in else cur[1]
            self.world_points[idx] = (new_x, new_y)
            if len(self.world_points)==4:
                xs = [p[0] for p in self.world_points]
                ys = [p[1] for p in self.world_points]
                self.world_width = max(xs)-min(xs)
                self.world_height = max(ys)-min(ys)
            if self.show_bev_live:
                self.update_bev_live_preview()
        except:
            print("❌ 输入无效")

    def reset_points(self):
        self.src_points = []
        self.line_points = []
        self.lane_lines = []
        self.lane_temp_points = []
        margin = 50
        self.src_points = [(margin,margin), (self.w-margin,margin),
                           (self.w-margin,self.h-margin), (margin,self.h-margin)]
        self.update_default_world_points()
        self.line_points = [(OFFSET,LINE_Y), (self.w-OFFSET,LINE_Y)]
        print("✅ 已重置所有点")
        if self.show_bev_live:
            self.update_bev_live_preview()

    # ------------------- BEV 预览相关（保持原始坐标） -------------------
    def polygon_area(self, points):
        area = 0
        n = len(points)
        for i in range(n):
            x1,y1 = points[i]
            x2,y2 = points[(i+1)%n]
            area += x1*y2 - x2*y1
        return area/2

    def add_reference_grid(self, img, w_w, w_h, scale):
        h,w = img.shape[:2]
        step = max(1, int(min(w_w,w_h)/5))
        px = max(10, int(step*scale))
        for x in range(0,w,px):
            cv2.line(img, (x,0), (x,h), (0,255,0),1)
            if x+5<w:
                cv2.putText(img, f"{int(x/scale)}m", (x+2,20), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
        for y in range(0,h,px):
            cv2.line(img, (0,y), (w,y), (0,255,0),1)
            if y+10<h:
                cv2.putText(img, f"{int(y/scale)}m", (2,y+10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
        return img


    ##################BEV视图############################
    def preview_bird_eye_view(self):
        if len(self.src_points) != 4:
            print("❌ 需要4个透视点")
            return
        try:
            src = np.array(self.src_points, dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)
            if self.polygon_area(src)*self.polygon_area(dst) < 0:
                dst = dst[::-1]
            M = cv2.getPerspectiveTransform(src, dst)
            w_w = max(dst[:,0])-min(dst[:,0])
            w_h = max(dst[:,1])-min(dst[:,1])
            target = 800
            scale = target / max(w_w,w_h) if w_w>0 else 20
            scale = max(10,min(100,scale))
            w_dst = max(100, int(w_w*scale))
            h_dst = max(100, int(w_h*scale))
            bev = cv2.warpPerspective(self.frame, M, (w_dst, h_dst))
            bev = self.add_reference_grid(bev, w_w, w_h, scale)
            cv2.imshow("BEV Preview", bev)
            cv2.waitKey(0)
            cv2.destroyWindow("BEV Preview")
        except Exception as e:
            print(f"❌ BEV预览失败: {e}")

    def toggle_bev_live_preview(self):
        if not self.show_bev_live:
            if len(self.src_points)==4:
                self.show_bev_live = True
                self.create_bev_preview_window()
                print("✅ 实时BEV预览已开启")
            else:
                print("❌ 需要4个透视点")
        else:
            self.close_bev_preview()
            print("✅ 实时BEV预览已关闭")

    def create_bev_preview_window(self):
        self.bev_window = "BEV Live Preview"
        cv2.namedWindow(self.bev_window)
        self.update_bev_live_preview()

    def update_bev_live_preview(self):
        if not self.show_bev_live or self.bev_window is None:
            return
        if len(self.src_points)!=4:
            return
        try:
            src = np.array(self.src_points[:4], dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)
            # 保证方向一致
            if self.polygon_area(src) * self.polygon_area(dst) < 0:
                dst = dst[::-1]
            # ---------- 计算真实尺度 ----------
            world_width = max(dst[:, 0]) - min(dst[:, 0])
            world_height = max(dst[:, 1]) - min(dst[:, 1])
            PIX_PER_METER = 10  # ⭐建议固定，不要动态scale
            bev_w = int(world_width * PIX_PER_METER)
            bev_h = int(world_height * PIX_PER_METER)
            bev_w = max(200, bev_w)
            bev_h = max(200, bev_h)
            # ---------- 目标像素坐标 ----------
            dst_pixels = np.array([
                [0, 0],
                [bev_w, 0],
                [bev_w, bev_h],
                [0, bev_h]
            ], dtype=np.float32)
            # ---------- 透视变换 ----------
            M = cv2.getPerspectiveTransform(src, dst_pixels)
            bev = cv2.warpPerspective(
                self.frame,
                M,
                (bev_w, bev_h)
            )
            # ---------- 加网格 ----------
            bev = self.add_reference_grid(
                bev,
                world_width,
                world_height,
                PIX_PER_METER
            )

            cv2.imshow(self.bev_window, bev)
            cv2.waitKey(1)
        except:
            pass

    def close_bev_preview(self):
        if self.bev_window is not None:
            self.show_bev_live = False
            try:
                cv2.destroyWindow(self.bev_window)
            except:
                pass
            self.bev_window = None



######################原始版本##############################################
class InteractiveCalibrationQurak:
    def __init__(self, frame, src_points=None, world_points=None, line_points=None):
        self.frame = frame.copy()
        self.canvas = self.frame.copy()
        self.h, self.w = frame.shape[:2]
        self.line_thickness = 1  # 统一线条粗细

        # 如果传入了预设点则使用，否则使用配置文件中的默认值
        if src_points is not None and world_points is not None:
            self.src_points = list(src_points)
            self.world_points = list(world_points)
        else:
            # 从 config 读取默认值（如果没有则使用内部默认）
            self.src_points = SOURCE_POINTS  # 图像上的点
            self.world_points = TARGET_POINTS  # 世界坐标点

        # 计数线：优先使用传入的，否则使用配置文件中的
        if line_points is not None:
            self.line_points = list(line_points)
        else:
            self.line_points = [(OFFSET, LINE_Y), (self.w - OFFSET, LINE_Y)]

        # 以下属性保持不变
        self.lane_lines = []
        self.lane_calib_mode = False
        self.dragging_idx = -1
        self.active_mode = "src"
        self.hover_idx = -1
        self.mouse_pos = (0, 0)
        self.world_width = WIDTH   # 米（用于显示）
        self.world_height = HEIGHT
        self.bev_window = None
        self.show_bev_live = False
        self.window_name = "Interactive Calibration"

    def edit_all_world_coords(self):
        """一次性编辑四个点的世界坐标（X, Y）"""
        if len(self.world_points) != 4:
            print("❌ 需要先有4个透视点才能编辑世界坐标")
            return

        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            root.title("编辑所有点的世界坐标 (X, Y 米)")
            root.geometry("400x300")
            root.attributes('-topmost', True)

            # 存储输入框变量的列表
            entries = []

            # 标题行
            ttk.Label(root, text="点序号", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
            ttk.Label(root, text="X (米)", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
            ttk.Label(root, text="Y (米)", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5, pady=5)

            for i, (wx, wy) in enumerate(self.world_points):
                label = ttk.Label(root, text=f"点 {i+1}")
                label.grid(row=i+1, column=0, padx=5, pady=5)

                var_x = tk.StringVar(value=f"{wx:.2f}")
                var_y = tk.StringVar(value=f"{wy:.2f}")
                entry_x = ttk.Entry(root, textvariable=var_x, width=10)
                entry_y = ttk.Entry(root, textvariable=var_y, width=10)
                entry_x.grid(row=i+1, column=1, padx=5, pady=5)
                entry_y.grid(row=i+1, column=2, padx=5, pady=5)
                entries.append((var_x, var_y))

            def on_confirm():
                try:
                    new_points = []
                    for var_x, var_y in entries:
                        x = float(var_x.get())
                        y = float(var_y.get())
                        new_points.append((x, y))
                    # 更新世界点
                    self.world_points = new_points
                    # 重新计算世界范围（用于显示）
                    xs = [p[0] for p in self.world_points]
                    ys = [p[1] for p in self.world_points]
                    self.world_width = max(xs) - min(xs)
                    self.world_height = max(ys) - min(ys)
                    print(f"✅ 世界坐标已更新为: {self.world_points}")
                    print(f"📐 世界范围: {self.world_width:.2f}m x {self.world_height:.2f}m")
                    root.destroy()
                    self.update_display()   # 刷新主画面
                    if self.show_bev_live:
                        self.update_bev_live_preview()
                except ValueError:
                    tk.messagebox.showerror("错误", "请输入有效的数字")

            def on_cancel():
                root.destroy()

            btn_frame = tk.Frame(root)
            btn_frame.grid(row=5, column=0, columnspan=3, pady=15)
            ttk.Button(btn_frame, text="确定", command=on_confirm).pack(side=tk.LEFT, padx=10)
            ttk.Button(btn_frame, text="取消", command=on_cancel).pack(side=tk.LEFT, padx=10)

            root.mainloop()

        except Exception as e:
            print(f"UI 对话框失败: {e}")
            # 回退到逐个控制台输入
            print("改用控制台逐个编辑...")
            for i in range(len(self.world_points)):
                self.edit_point_world_coord(i)

    def run(self):
        """运行交互标定流程"""
        # 初始化默认点
        if len(self.src_points) == 0:
            margin = 50
            self.src_points = [
                (margin, margin),
                (self.w - margin, margin),
                (self.w - margin, self.h - margin),
                (margin, self.h - margin)
            ]
            self.update_default_world_points()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("🎯 交互式标定工具使用说明")
        print("=" * 60)
        print("【鼠标操作】")
        print("  • 左键拖动点 → 移动点位置")
        print("  • 右键点击点 → 编辑该点的世界坐标")
        print("  • 鼠标悬停 → 显示坐标信息")
        print("\n【键盘命令】")
        print("  • [1] 编辑透视点（红色）")
        print("  • [2] 编辑计数线（蓝色）")
        print("  • [3] 编辑车道线（绿色）- 点击画线，自动拟合为直线")
        print("  • [C] 编辑所有点的世界坐标")
        print("  • [R] 重置所有点")
        print("  • [U] 撤销最后一次点击")
        print("  • [P] 预览俯视图变换效果")
        print("  • [L] 实时BEV预览模式")
        print("  • [ENTER] 完成标定并保存")
        print("  • [ESC] 取消标定")
        print("=" * 60)

        while True:
            self.update_display()
            cv2.imshow(self.window_name, self.canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # Enter
                if len(self.src_points) == 4 and len(self.line_points) == 2:
                    break
                else:
                    print(f"❌ 需要4个透视点（当前{len(self.src_points)}）和2个计数线点（当前{len(self.line_points)}）")

            elif key == 27:  # ESC
                self.close_bev_preview()
                return None

            elif key == ord('1'):
                self.active_mode = "src"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑透视点（红色）")

            elif key == ord('2'):
                self.active_mode = "line"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑计数线（蓝色）")

            elif key == ord('3'):
                self.active_mode = "lane"
                self.lane_calib_mode = True
                print("✅ 当前模式：编辑车道线（绿色）")
                print("   提示：在图像上点击多个点，系统会自动拟合为垂直线")

            elif key == ord('u') or key == ord('U'):
                if self.lane_calib_mode and len(self.lane_temp_points) > 0:
                    self.lane_temp_points.pop()
                    print(f"🗑️ 撤销一点，剩余 {len(self.lane_temp_points)} 个点")
                    if len(self.lane_temp_points) >= 2:
                        self.fit_lane_line()
                elif self.active_mode == "src" and len(self.src_points) > 0:
                    self.src_points.pop()
                    print(f"🗑️ 撤销透视点，剩余 {len(self.src_points)} 个点")
                elif self.active_mode == "line" and len(self.line_points) > 0:
                    self.line_points.pop()
                    print(f"🗑️ 撤销计数线点，剩余 {len(self.line_points)} 个点")

            # elif key == ord('w') or key == ord('W'):
            #     self.set_world_range()

            elif key == ord('r') or key == ord('R'):
                self.reset_points()

            elif key == ord('p') or key == ord('P'):
                self.preview_bird_eye_view()

            elif key == ord('l') or key == ord('L'):
                self.toggle_bev_live_preview()

            # elif key == ord('s') or key == ord('S'):
            #     self.set_road_preset()

        self.close_bev_preview()
        cv2.destroyWindow(self.window_name)

        # 返回标定数据（增加车道线列表）
        src_array = np.array(self.src_points, dtype=np.float32)
        world_array = np.array(self.world_points, dtype=np.float32)
        return src_array, world_array, self.line_points[0], self.line_points[1], self.lane_lines

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        self.mouse_pos = (x, y)

        # 车道标定模式特殊处理
        if self.lane_calib_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # 添加车道标定点
                if not hasattr(self, 'lane_temp_points'):
                    self.lane_temp_points = []
                self.lane_temp_points.append((x, y))
                print(f"📍 车道标定点 {len(self.lane_temp_points)}: ({x}, {y})")

                # 如果有2个及以上点，拟合直线
                if len(self.lane_temp_points) >= 2:
                    self.fit_lane_line()
                self.update_display()
            return

        # 获取当前正在编辑的点集
        points = self.src_points if self.active_mode == "src" else self.line_points

        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self.get_hover_point(x, y, points)

        elif event == cv2.EVENT_LBUTTONDOWN:
            idx = self.get_hover_point(x, y, points)
            if idx >= 0:
                self.dragging_idx = idx
            else:
                if self.active_mode == "src" and len(points) < 4:
                    points.append((x, y))
                    if len(points) == 4:
                        self.update_default_world_points()
                    print(f"✅ 添加透视点 {len(points)}/4: ({x}, {y})")
                elif self.active_mode == "line" and len(points) < 2:
                    points.append((x, y))
                    print(f"✅ 添加计数线点 {len(points)}/2: ({x}, {y})")

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self.get_hover_point(x, y, self.src_points)
            if idx >= 0 and self.active_mode == "src":
                self.edit_point_world_coord(idx)

    def get_hover_point(self, x, y, points):
        """检查鼠标是否悬停在某个点上"""
        for i, pt in enumerate(points):
            if abs(x - pt[0]) < 8 and abs(y - pt[1]) < 8:
                return i
        return -1

    def fit_lane_line(self):
        if not hasattr(self, 'lane_temp_points') or len(self.lane_temp_points) < 2:
            return

        points = np.array(self.lane_temp_points)
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        x_std = np.std(points[:, 0])
        if x_std < 10:
            x_center = int(np.mean(points[:, 0]))
            lane_line = (x_center, int(y_min), x_center, int(y_max))
        else:
            x = points[:, 0].reshape(-1, 1)
            y = points[:, 1].reshape(-1, 1)
            A = np.hstack([y, np.ones_like(y)])
            params = np.linalg.lstsq(A, x, rcond=None)[0]
            a, b = params[0][0], params[1][0]
            x1 = int(a * y_min + b)
            x2 = int(a * y_max + b)
            lane_line = (x1, int(y_min), x2, int(y_max))

        self.lane_lines.append(lane_line)
        print(
            f"✅ 已添加车道线 {len(self.lane_lines)}: ({lane_line[0]},{lane_line[1]}) -> ({lane_line[2]},{lane_line[3]})")
        self.lane_temp_points = []
        self.update_display()

    def update_display(self):
        """更新显示画面"""
        self.canvas = self.frame.copy()

        # 根据当前模式绘制不同颜色的点
        if self.active_mode == "src":
            self.draw_points(self.src_points, (0, 0, 255), "src")
            self.draw_points(self.line_points, (255, 0, 0), "line", alpha=0.5)
        elif self.active_mode == "line":
            self.draw_points(self.src_points, (0, 0, 255), "src", alpha=0.5)
            self.draw_points(self.line_points, (255, 0, 0), "line")
        else:  # lane mode
            self.draw_points(self.src_points, (0, 0, 255), "src", alpha=0.5)
            self.draw_points(self.line_points, (255, 0, 0), "line", alpha=0.5)

        # ========== 优化车道线可视化 ==========
        # 1. 可选：半透明填充车道区域（需要额外计算，若不需要可跳过）
        if len(self.lane_lines) >= 2:
            overlay = self.canvas.copy()
            for i in range(len(self.lane_lines) - 1):
                # 获取相邻两条车道线的四个端点，构成一个封闭多边形
                x1a, y1a, x2a, y2a = self.lane_lines[i]
                x1b, y1b, x2b, y2b = self.lane_lines[i + 1]
                pts = np.array([[x1a, y1a], [x2a, y2a], [x2b, y2b], [x1b, y1b]], np.int32)
                # 半透明填充（紫色系，低透明度）
                cv2.fillPoly(overlay, [pts], (100, 0, 255))  # BGR 紫色
            # 混合原图与填充层
            cv2.addWeighted(overlay, 0.2, self.canvas, 0.8, 0, self.canvas)

        # 2. 绘制虚线车道线 + 端点装饰
        for i, (x1, y1, x2, y2) in enumerate(self.lane_lines):
            # 虚线样式：计算线段长度和分段数
            length = int(np.hypot(x2 - x1, y2 - y1))
            dash_len = 20  # 虚线每段长度（像素）
            solid_len = 25
            if length > dash_len + solid_len:
                # 计算单位方向向量
                dx = (x2 - x1) / length
                dy = (y2 - y1) / length
                current = 0
                while current < length:
                    start_pt = (int(x1 + dx * current), int(y1 + dy * current))
                    end_pt = (int(x1 + dx * min(current + dash_len, length)),
                              int(y1 + dy * min(current + dash_len, length)))
                    cv2.line(self.canvas, start_pt, end_pt, (0, 255, 255), 3, cv2.LINE_AA)  # 黄色虚线
                    current += dash_len + solid_len
            else:
                # 线段太短时直接画实线
                cv2.line(self.canvas, (x1, y1), (x2, y2), (0, 255, 255), 3, cv2.LINE_AA)

            # 在车道线两个端点添加圆点（可选）
            cv2.circle(self.canvas, (x1, y1), 5, (0, 255, 255), -1)
            cv2.circle(self.canvas, (x2, y2), 5, (0, 255, 255), -1)

            # 绘制车道编号（带背景框，更醒目）
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            text = f"Lane {i + 1}"
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                         font_scale, thickness)
            # 背景框位置（文本框居中）
            box_x = mid_x - text_w // 2 - 5
            box_y = mid_y - text_h // 2 - 5
            # 绘制半透明黑色背景圆角矩形（简单矩形也可）
            cv2.rectangle(self.canvas, (box_x, box_y), (box_x + text_w + 10, box_y + text_h + 10),
                          (0, 0, 0), -1)
            # 绘制文字
            cv2.putText(self.canvas, text, (mid_x - text_w // 2, mid_y + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        # 绘制临时车道标定点
        if hasattr(self, 'lane_temp_points') and self.lane_temp_points:
            for pt in self.lane_temp_points:
                cv2.circle(self.canvas, pt, 5, (0, 255, 255), -1)
                cv2.circle(self.canvas, pt, 7, (0, 255, 255), 2)

            # 如果临时点有2个以上，绘制拟合直线预览
            if len(self.lane_temp_points) >= 2:
                points = np.array(self.lane_temp_points)
                x_std = np.std(points[:, 0])
                if x_std < 10:
                    x_center = int(np.mean(points[:, 0]))
                    cv2.line(self.canvas, (x_center, 0), (x_center, self.h), (0, 255, 255), 2)
                else:
                    x = points[:, 0].reshape(-1, 1)
                    y = points[:, 1].reshape(-1, 1)
                    A = np.hstack([y, np.ones_like(y)])
                    params = np.linalg.lstsq(A, x, rcond=None)[0]
                    a, b = params[0][0], params[1][0]
                    x1, x2 = int(a * 0 + b), int(a * self.h + b)
                    cv2.line(self.canvas, (x1, 0), (x2, self.h), (0, 255, 255), 2)

        # 绘制计数线
        if len(self.line_points) == 2:
            cv2.line(self.canvas, self.line_points[0], self.line_points[1], (255, 0, 0), 2)
            self.draw_arrow(self.line_points[0], self.line_points[1], (255, 0, 0))

        # 绘制透视四边形
        if len(self.src_points) == 4:
            pts = np.array(self.src_points, dtype=np.int32)
            cv2.polylines(self.canvas, [pts], True, (0, 255, 0), self.line_thickness)

            # 绘制连接箭头显示点序
            for i in range(4):
                j = (i + 1) % 4
                self.draw_arrow(self.src_points[i], self.src_points[j], (0, 255, 0), 1, tip_length=0.2)

            # 在中心显示世界范围信息
            # center_x = sum(p[0] for p in self.src_points) // 4
            # center_y = sum(p[1] for p in self.src_points) // 4
            # info_text = f"World: {self.world_width:.1f}m x {self.world_height:.1f}m"
            # cv2.putText(self.canvas, info_text, (center_x - 100, center_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示提示信息
        if self.lane_calib_mode:
            mode_text = f"Lane Mode ({len(self.lane_lines)} lanes, {len(self.lane_temp_points) if hasattr(self, 'lane_temp_points') else 0} temp points)"
            cv2.putText(self.canvas, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            mode_text = "Perspective Mode" if self.active_mode == "src" else "Count Line Mode"
            cv2.putText(self.canvas, f"Mode: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        bev_status = "BEV Live: ON" if self.show_bev_live else "BEV Live: OFF"
        cv2.putText(self.canvas, bev_status, (self.w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.show_bev_live else (0, 0, 255), 1)

        # 显示鼠标当前位置的像素坐标（固定右上角）
        mouse_pos_text = f"Cursor: ({self.mouse_pos[0]}, {self.mouse_pos[1]})"
        cv2.putText(self.canvas, mouse_pos_text, (self.w - 330, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 显示鼠标悬停点的坐标
        if self.hover_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode == "src" else self.line_points
            if self.hover_idx < len(points):
                pt = points[self.hover_idx]
                if self.active_mode == "src" and self.hover_idx < len(self.world_points):
                    world = self.world_points[self.hover_idx]
                    coord_text = f"P{self.hover_idx + 1}: img({pt[0]},{pt[1]}) world({world[0]:.1f},{world[1]:.1f})m"
                else:
                    coord_text = f"L{self.hover_idx + 1}: ({pt[0]},{pt[1]})"
                cv2.putText(self.canvas, coord_text, (self.mouse_pos[0] + 10, self.mouse_pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(self.canvas, coord_text, (self.mouse_pos[0] + 9, self.mouse_pos[1] - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 拖动更新点位置
        if self.dragging_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode == "src" else self.line_points
            if self.dragging_idx < len(points):
                x, y = self.mouse_pos
                x = max(0, min(self.w, x))
                y = max(0, min(self.h, y))
                points[self.dragging_idx] = (x, y)
                if self.active_mode == "src" and len(self.src_points) == 4:
                    pass
                    # self.update_default_world_points()
                if self.show_bev_live and self.active_mode == "src" and len(self.src_points) == 4:
                    self.update_bev_live_preview()

    def set_road_preset(self):
        """快速设置为道路场景常用参数"""
        self.world_width = 3.5
        self.world_height = 20.0
        self.update_default_world_points()
        print("✅ 已设置为道路标准参数（宽3.5m x 长20m）")
        if self.show_bev_live:
            self.update_bev_live_preview()

    def update_default_world_points(self):
        """根据世界范围更新四个点的世界坐标（矩形）"""
        if len(self.src_points) == 4:
            self.world_points = [
                (0, 0),
                (self.world_width, 0),
                (self.world_width, self.world_height),
                (0, self.world_height)
            ]

    def draw_points(self, points, color, point_type, alpha=1.0):
        """绘制点集"""
        for i, pt in enumerate(points):
            cv2.circle(self.canvas, pt, 6, color, -1)
            cv2.circle(self.canvas, pt, 8, color, 2)
            cv2.putText(self.canvas, str(i + 1), (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.hover_idx == i and not self.lane_calib_mode and \
                    ((self.active_mode == "src" and point_type == "src") or
                     (self.active_mode == "line" and point_type == "line")):
                cv2.circle(self.canvas, pt, 12, (255, 255, 0), 2)

    def draw_arrow(self, start, end, color, thickness=2, tip_length=0.3):
        """绘制箭头"""
        start_pt = np.array(start)
        end_pt = np.array(end)
        direction = end_pt - start_pt
        length = np.linalg.norm(direction)
        if length < 1:
            return
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])
        arrow_size = min(20, length * tip_length)
        tip_point = end_pt - direction * arrow_size
        left_point = tip_point + perpendicular * (arrow_size * 0.5)
        right_point = tip_point - perpendicular * (arrow_size * 0.5)
        pts = np.array([end_pt, left_point, tip_point, right_point], dtype=np.int32)
        cv2.fillPoly(self.canvas, [pts], color)

    def set_world_range(self):
        """设置世界坐标范围（修复对话框不弹出问题）"""
        try:
            import tkinter as tk
            from tkinter import simpledialog

            # 创建根窗口并隐藏
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)  # 确保在最前

            # 使用 simpledialog 独立弹窗，不依赖 root 显示
            new_width = simpledialog.askfloat("世界宽度", "请输入宽度（米）",
                                              initialvalue=self.world_width,
                                              parent=root)
            if new_width is not None:
                self.world_width = new_width

            new_height = simpledialog.askfloat("世界高度", "请输入长度（米）",
                                               initialvalue=self.world_height,
                                               parent=root)
            if new_height is not None:
                self.world_height = new_height

            root.destroy()
            self.update_default_world_points()
            print(f"✅ 已更新世界范围: 宽={self.world_width:.1f}m, 长={self.world_height:.1f}m")

        except Exception as e:
            print(f"UI 对话框失败: {e}, 改用控制台输入")
            # 回退到控制台（见下方）
            self._set_world_range_console()

    def _set_world_range_console(self):
        """控制台备用输入"""
        print(f"\n当前世界范围: 宽={self.world_width:.1f}m, 长={self.world_height:.1f}m")
        try:
            w_input = input(f"请输入宽度（米）[默认{self.world_width:.1f}]: ").strip()
            if w_input:
                self.world_width = float(w_input)
            h_input = input(f"请输入长度（米）[默认{self.world_height:.1f}]: ").strip()
            if h_input:
                self.world_height = float(h_input)
            self.update_default_world_points()
            print(f"✅ 已更新世界范围: {self.world_width:.1f}m x {self.world_height:.1f}m")
        except ValueError:
            print("❌ 输入无效，保持原值")

    def edit_point_world_coord(self, idx):
        """编辑单个点的世界坐标"""
        if idx >= len(self.world_points):
            return

        current = self.world_points[idx]
        print(f"\n编辑点 {idx + 1} 的世界坐标（当前: x={current[0]:.2f}m, y={current[1]:.2f}m）")

        try:
            x_input = input(f"请输入新的X坐标（米）[默认{current[0]:.2f}]: ").strip()
            y_input = input(f"请输入新的Y坐标（米）[默认{current[1]:.2f}]: ").strip()

            new_x = float(x_input) if x_input else current[0]
            new_y = float(y_input) if y_input else current[1]

            self.world_points[idx] = (new_x, new_y)
            print(f"✅ 已更新点{idx + 1}世界坐标为: ({new_x:.2f}, {new_y:.2f})")

            if len(self.world_points) == 4:
                xs = [p[0] for p in self.world_points]
                ys = [p[1] for p in self.world_points]
                self.world_width = max(xs) - min(xs)
                self.world_height = max(ys) - min(ys)
                print(f"📐 世界范围已更新为: {self.world_width:.2f}m x {self.world_height:.2f}m")

            if self.show_bev_live:
                self.update_bev_live_preview()

        except ValueError:
            print("❌ 输入无效，未更新")

    def reset_points(self):
        """重置所有点"""
        self.src_points = []
        self.line_points = []
        self.lane_lines = []
        if hasattr(self, 'lane_temp_points'):
            self.lane_temp_points = []
        self.world_points = []

        margin = 50
        self.src_points = [
            (margin, margin),
            (self.w - margin, margin),
            (self.w - margin, self.h - margin),
            (margin, self.h - margin)
        ]
        self.update_default_world_points()
        print("✅ 已重置所有点")
        if self.show_bev_live:
            self.update_bev_live_preview()

    def polygon_area(self, points):
        """计算多边形有向面积"""
        area = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2

    def add_reference_grid(self, image, world_width, world_height, scale):
        """添加参考网格线"""
        img = image.copy()
        h, w = img.shape[:2]

        grid_spacing = max(1, int(min(world_width, world_height) / 5))
        grid_pixels = max(10, int(grid_spacing * scale))

        for x in range(0, w, grid_pixels):
            cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
            if x + 5 < w:
                cv2.putText(img, f"{int(x / scale)}m", (x + 2, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for y in range(0, h, grid_pixels):
            cv2.line(img, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)
            if y + 10 < h:
                cv2.putText(img, f"{int(y / scale)}m", (2, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return img

    def preview_bird_eye_view(self):
        """预览俯视图变换效果"""
        if len(self.src_points) != 4:
            print("❌ 需要4个透视点才能预览俯视图")
            return

        try:
            src = np.array(self.src_points, dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)

            src_area = self.polygon_area(src)
            dst_area = self.polygon_area(dst)

            if src_area * dst_area < 0:
                print("⚠️ 警告：源点和目标点的顺序不一致，正在自动修正...")
                dst = dst[::-1]
                self.world_points = [(p[0], p[1]) for p in dst]

            M = cv2.getPerspectiveTransform(src, dst)

            world_width = max(dst[:, 0]) - min(dst[:, 0])
            world_height = max(dst[:, 1]) - min(dst[:, 1])

            target_display_size = 800
            if world_width > 0:
                scale = target_display_size / max(world_width, world_height)
            else:
                scale = 20

            scale = max(10, min(100, scale))

            dst_width = max(100, int(world_width * scale)) if world_width > 0 else 800
            dst_height = max(100, int(world_height * scale)) if world_height > 0 else 600

            bird_view = cv2.warpPerspective(self.frame, M, (dst_width, dst_height))
            bird_view = self.add_reference_grid(bird_view, world_width, world_height, scale)

            cv2.putText(bird_view, f"BEV Preview (Scale: {scale:.1f}px/m)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bird_view, f"Region: {world_width:.2f}m x {world_height:.2f}m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 在BEV图上绘制车道线
            if self.lane_lines:
                M_inv = cv2.getPerspectiveTransform(dst, src)
                for lane_line in self.lane_lines:
                    x1, y1, x2, y2 = lane_line
                    # 将图像上的车道线转换到BEV视图
                    pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
                    world_pts = cv2.perspectiveTransform(pts, M)
                    if len(world_pts) >= 2:
                        pt1 = (int(world_pts[0][0][0]), int(world_pts[0][0][1]))
                        pt2 = (int(world_pts[1][0][0]), int(world_pts[1][0][1]))
                        # 调整到BEV图像坐标
                        scale_x = dst_width / world_width if world_width > 0 else 1
                        scale_y = dst_height / world_height if world_height > 0 else 1
                        pt1_bev = (int(pt1[0] * scale_x), int(pt1[1] * scale_y))
                        pt2_bev = (int(pt2[0] * scale_x), int(pt2[1] * scale_y))
                        cv2.line(bird_view, pt1_bev, pt2_bev, (0, 255, 255), 2)

            preview_name = "Bird's Eye View Preview (Press any key to close)"
            cv2.imshow(preview_name, bird_view)
            print("\n✅ BEV图已生成，按任意键关闭预览")
            cv2.waitKey(0)

            try:
                cv2.destroyWindow(preview_name)
            except:
                pass

        except cv2.error as e:
            print(f"❌ OpenCV透视变换失败: {e}")
        except Exception as e:
            print(f"❌ 预览失败: {e}")

    def toggle_bev_live_preview(self):
        """切换实时BEV预览模式"""
        if not self.show_bev_live:
            if len(self.src_points) == 4:
                self.show_bev_live = True
                self.create_bev_preview_window()
                print("✅ 实时BEV预览已开启")
            else:
                print("❌ 需要先选择4个透视点才能开启实时预览")
        else:
            self.close_bev_preview()
            print("✅ 实时BEV预览已关闭")

    def create_bev_preview_window(self):
        """创建实时BEV预览窗口"""
        self.bev_window = "BEV Live Preview (Auto-update)"
        cv2.namedWindow(self.bev_window)
        self.update_bev_live_preview()

    def update_bev_live_preview(self):
        if not self.show_bev_live or self.bev_window is None:
            return
        if len(self.src_points)!=4:
            return
        try:
            src = np.array(self.src_points[:4], dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)
            # 保证方向一致
            if self.polygon_area(src) * self.polygon_area(dst) < 0:
                dst = dst[::-1]
            # ---------- 计算真实尺度 ----------
            world_width = max(dst[:, 0]) - min(dst[:, 0])
            world_height = max(dst[:, 1]) - min(dst[:, 1])
            PIX_PER_METER = 10  # ⭐建议固定，不要动态scale
            bev_w = int(world_width * PIX_PER_METER)
            bev_h = int(world_height * PIX_PER_METER)
            bev_w = max(200, bev_w)
            bev_h = max(200, bev_h)
            # ---------- 目标像素坐标 ----------
            dst_pixels = np.array([
                [0, 0],
                [bev_w, 0],
                [bev_w, bev_h],
                [0, bev_h]
            ], dtype=np.float32)
            # ---------- 透视变换 ----------
            M = cv2.getPerspectiveTransform(src, dst_pixels)
            bev = cv2.warpPerspective(
                self.frame,
                M,
                (bev_w, bev_h)
            )
            # ---------- 加网格 ----------
            bev = self.add_reference_grid(
                bev,
                world_width,
                world_height,
                PIX_PER_METER
            )

            cv2.imshow(self.bev_window, bev)
            cv2.waitKey(1)
        except:
            pass

    def close_bev_preview(self):
        """关闭实时BEV预览窗口"""
        if self.bev_window is not None:
            self.show_bev_live = False
            try:
                cv2.destroyWindow(self.bev_window)
            except:
                pass
            self.bev_window = None
class InteractiveCalibration:
    """高交互式标定工具 - 支持拖动点、编辑坐标、实时预览"""
    def __init__(self, frame):
        """
        这里默认初始标定值在配置表中设置
        """
        self.frame = frame.copy()
        self.canvas = self.frame.copy()
        self.h, self.w = frame.shape[:2]
        self.line_thickness = 1  # 统一线条粗细

        # 标定点（4个透视点）
        self.src_points = SOURCE_POINTS  # 图像上的点 [(x,y), ...]
        self.world_points = TARGET_POINTS  # 世界坐标点 [(x_m, y_m), ...]

        # 计数线（2个点）
        self.line_points = [(OFFSET, LINE_Y),(self.w - OFFSET, LINE_Y)]

        # 车道线（多条垂直线）
        self.lane_lines = []  # 存储车道线的图像坐标 [(x1,y1, x2,y2), ...]
        self.lane_calib_mode = False  # 车道标定模式

        # 交互状态
        self.dragging_idx = -1  # 正在拖动的点索引
        self.active_mode = "src"  # "src", "line", "lane"
        self.hover_idx = -1  # 鼠标悬停的点索引
        self.mouse_pos = (0, 0)

        # 世界坐标范围
        self.world_width = WIDTH  # 米
        self.world_height = HEIGHT  # 米

        # BEV预览窗口
        self.bev_window = None
        self.show_bev_live = False

        # 窗口名称
        self.window_name = "Interactive Calibration"

    def run(self):
        """运行交互标定流程"""
        # 初始化默认点
        if len(self.src_points) == 0:
            margin = 50
            self.src_points = [
                (margin, margin),
                (self.w - margin, margin),
                (self.w - margin, self.h - margin),
                (margin, self.h - margin)
            ]
            self.update_default_world_points()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("🎯 交互式标定工具使用说明")
        print("=" * 60)
        print("【鼠标操作】")
        print("  • 左键拖动点 → 移动点位置")
        print("  • 右键点击点 → 编辑该点的世界坐标")
        print("  • 鼠标悬停 → 显示坐标信息")
        print("\n【键盘命令】")
        print("  • [1] 编辑透视点（红色）")
        print("  • [2] 编辑计数线（蓝色）")
        print("  • [3] 编辑车道线（绿色）- 点击画线，自动拟合为直线")
        print("  • [W] 设置世界范围（宽/高）")
        print("  • [R] 重置所有点")
        print("  • [U] 撤销最后一次点击")
        print("  • [P] 预览俯视图变换效果")
        print("  • [L] 实时BEV预览模式")
        print("  • [S] 道路标准预设")
        print("  • [ENTER] 完成标定并保存")
        print("  • [ESC] 取消标定")
        print("=" * 60)

        while True:
            self.update_display()
            cv2.imshow(self.window_name, self.canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # Enter
                if len(self.src_points) == 4 and len(self.line_points) == 2:
                    break
                else:
                    print(f"❌ 需要4个透视点（当前{len(self.src_points)}）和2个计数线点（当前{len(self.line_points)}）")

            elif key == 27:  # ESC
                self.close_bev_preview()
                return None

            elif key == ord('1'):
                self.active_mode = "src"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑透视点（红色）")

            elif key == ord('2'):
                self.active_mode = "line"
                self.lane_calib_mode = False
                print("✅ 当前模式：编辑计数线（蓝色）")

            elif key == ord('3'):
                self.active_mode = "lane"
                self.lane_calib_mode = True
                print("✅ 当前模式：编辑车道线（绿色）")
                print("   提示：在图像上点击多个点，系统会自动拟合为垂直线")

            elif key == ord('u') or key == ord('U'):
                if self.lane_calib_mode and len(self.lane_temp_points) > 0:
                    self.lane_temp_points.pop()
                    print(f"🗑️ 撤销一点，剩余 {len(self.lane_temp_points)} 个点")
                    if len(self.lane_temp_points) >= 2:
                        self.fit_lane_line()
                elif self.active_mode == "src" and len(self.src_points) > 0:
                    self.src_points.pop()
                    print(f"🗑️ 撤销透视点，剩余 {len(self.src_points)} 个点")
                elif self.active_mode == "line" and len(self.line_points) > 0:
                    self.line_points.pop()
                    print(f"🗑️ 撤销计数线点，剩余 {len(self.line_points)} 个点")

            elif key == ord('w') or key == ord('W'):
                self.set_world_range()

            elif key == ord('r') or key == ord('R'):
                self.reset_points()

            elif key == ord('p') or key == ord('P'):
                self.preview_bird_eye_view()

            elif key == ord('l') or key == ord('L'):
                self.toggle_bev_live_preview()

            elif key == ord('s') or key == ord('S'):
                self.set_road_preset()

        self.close_bev_preview()
        cv2.destroyWindow(self.window_name)

        # 返回标定数据（增加车道线列表）
        src_array = np.array(self.src_points, dtype=np.float32)
        world_array = np.array(self.world_points, dtype=np.float32)
        return src_array, world_array, self.line_points[0], self.line_points[1], self.lane_lines

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        self.mouse_pos = (x, y)

        # 车道标定模式特殊处理
        if self.lane_calib_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # 添加车道标定点
                if not hasattr(self, 'lane_temp_points'):
                    self.lane_temp_points = []
                self.lane_temp_points.append((x, y))
                print(f"📍 车道标定点 {len(self.lane_temp_points)}: ({x}, {y})")

                # 如果有2个及以上点，拟合直线
                if len(self.lane_temp_points) >= 2:
                    self.fit_lane_line()
                self.update_display()
            return

        # 获取当前正在编辑的点集
        points = self.src_points if self.active_mode == "src" else self.line_points

        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self.get_hover_point(x, y, points)

        elif event == cv2.EVENT_LBUTTONDOWN:
            idx = self.get_hover_point(x, y, points)
            if idx >= 0:
                self.dragging_idx = idx
            else:
                if self.active_mode == "src" and len(points) < 4:
                    points.append((x, y))
                    if len(points) == 4:
                        self.update_default_world_points()
                    print(f"✅ 添加透视点 {len(points)}/4: ({x}, {y})")
                elif self.active_mode == "line" and len(points) < 2:
                    points.append((x, y))
                    print(f"✅ 添加计数线点 {len(points)}/2: ({x}, {y})")

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self.get_hover_point(x, y, self.src_points)
            if idx >= 0 and self.active_mode == "src":
                self.edit_point_world_coord(idx)

    def get_hover_point(self, x, y, points):
        """检查鼠标是否悬停在某个点上"""
        for i, pt in enumerate(points):
            if abs(x - pt[0]) < 8 and abs(y - pt[1]) < 8:
                return i
        return -1

    def fit_lane_line(self):
        if not hasattr(self, 'lane_temp_points') or len(self.lane_temp_points) < 2:
            return

        points = np.array(self.lane_temp_points)
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        x_std = np.std(points[:, 0])
        if x_std < 10:
            x_center = int(np.mean(points[:, 0]))
            lane_line = (x_center, int(y_min), x_center, int(y_max))
        else:
            x = points[:, 0].reshape(-1, 1)
            y = points[:, 1].reshape(-1, 1)
            A = np.hstack([y, np.ones_like(y)])
            params = np.linalg.lstsq(A, x, rcond=None)[0]
            a, b = params[0][0], params[1][0]
            x1 = int(a * y_min + b)
            x2 = int(a * y_max + b)
            lane_line = (x1, int(y_min), x2, int(y_max))

        self.lane_lines.append(lane_line)
        print(
            f"✅ 已添加车道线 {len(self.lane_lines)}: ({lane_line[0]},{lane_line[1]}) -> ({lane_line[2]},{lane_line[3]})")
        self.lane_temp_points = []
        self.update_display()

    def update_display(self):
        """更新显示画面"""
        self.canvas = self.frame.copy()

        # 根据当前模式绘制不同颜色的点
        if self.active_mode == "src":
            self.draw_points(self.src_points, (0, 0, 255), "src")
            self.draw_points(self.line_points, (255, 0, 0), "line", alpha=0.5)
        elif self.active_mode == "line":
            self.draw_points(self.src_points, (0, 0, 255), "src", alpha=0.5)
            self.draw_points(self.line_points, (255, 0, 0), "line")
        else:  # lane mode
            self.draw_points(self.src_points, (0, 0, 255), "src", alpha=0.5)
            self.draw_points(self.line_points, (255, 0, 0), "line", alpha=0.5)

        # ========== 优化车道线可视化 ==========
        # 1. 可选：半透明填充车道区域（需要额外计算，若不需要可跳过）
        if len(self.lane_lines) >= 2:
            overlay = self.canvas.copy()
            for i in range(len(self.lane_lines) - 1):
                # 获取相邻两条车道线的四个端点，构成一个封闭多边形
                x1a, y1a, x2a, y2a = self.lane_lines[i]
                x1b, y1b, x2b, y2b = self.lane_lines[i + 1]
                pts = np.array([[x1a, y1a], [x2a, y2a], [x2b, y2b], [x1b, y1b]], np.int32)
                # 半透明填充（紫色系，低透明度）
                cv2.fillPoly(overlay, [pts], (100, 0, 255))  # BGR 紫色
            # 混合原图与填充层
            cv2.addWeighted(overlay, 0.2, self.canvas, 0.8, 0, self.canvas)

        # 2. 绘制虚线车道线 + 端点装饰
        for i, (x1, y1, x2, y2) in enumerate(self.lane_lines):
            # 虚线样式：计算线段长度和分段数
            length = int(np.hypot(x2 - x1, y2 - y1))
            dash_len = 20  # 虚线每段长度（像素）
            solid_len = 25
            if length > dash_len + solid_len:
                # 计算单位方向向量
                dx = (x2 - x1) / length
                dy = (y2 - y1) / length
                current = 0
                while current < length:
                    start_pt = (int(x1 + dx * current), int(y1 + dy * current))
                    end_pt = (int(x1 + dx * min(current + dash_len, length)),
                              int(y1 + dy * min(current + dash_len, length)))
                    cv2.line(self.canvas, start_pt, end_pt, (0, 255, 255), 3, cv2.LINE_AA)  # 黄色虚线
                    current += dash_len + solid_len
            else:
                # 线段太短时直接画实线
                cv2.line(self.canvas, (x1, y1), (x2, y2), (0, 255, 255), 3, cv2.LINE_AA)

            # 在车道线两个端点添加圆点（可选）
            cv2.circle(self.canvas, (x1, y1), 5, (0, 255, 255), -1)
            cv2.circle(self.canvas, (x2, y2), 5, (0, 255, 255), -1)

            # 绘制车道编号（带背景框，更醒目）
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            text = f"Lane {i + 1}"
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                         font_scale, thickness)
            # 背景框位置（文本框居中）
            box_x = mid_x - text_w // 2 - 5
            box_y = mid_y - text_h // 2 - 5
            # 绘制半透明黑色背景圆角矩形（简单矩形也可）
            cv2.rectangle(self.canvas, (box_x, box_y), (box_x + text_w + 10, box_y + text_h + 10),
                          (0, 0, 0), -1)
            # 绘制文字
            cv2.putText(self.canvas, text, (mid_x - text_w // 2, mid_y + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        # 绘制临时车道标定点
        if hasattr(self, 'lane_temp_points') and self.lane_temp_points:
            for pt in self.lane_temp_points:
                cv2.circle(self.canvas, pt, 5, (0, 255, 255), -1)
                cv2.circle(self.canvas, pt, 7, (0, 255, 255), 2)

            # 如果临时点有2个以上，绘制拟合直线预览
            if len(self.lane_temp_points) >= 2:
                points = np.array(self.lane_temp_points)
                x_std = np.std(points[:, 0])
                if x_std < 10:
                    x_center = int(np.mean(points[:, 0]))
                    cv2.line(self.canvas, (x_center, 0), (x_center, self.h), (0, 255, 255), 2)
                else:
                    x = points[:, 0].reshape(-1, 1)
                    y = points[:, 1].reshape(-1, 1)
                    A = np.hstack([y, np.ones_like(y)])
                    params = np.linalg.lstsq(A, x, rcond=None)[0]
                    a, b = params[0][0], params[1][0]
                    x1, x2 = int(a * 0 + b), int(a * self.h + b)
                    cv2.line(self.canvas, (x1, 0), (x2, self.h), (0, 255, 255), 2)

        # 绘制计数线
        if len(self.line_points) == 2:
            cv2.line(self.canvas, self.line_points[0], self.line_points[1], (255, 0, 0), 2)
            self.draw_arrow(self.line_points[0], self.line_points[1], (255, 0, 0))

        # 绘制透视四边形
        if len(self.src_points) == 4:
            pts = np.array(self.src_points, dtype=np.int32)
            cv2.polylines(self.canvas, [pts], True, (0, 255, 0), self.line_thickness)

            # 绘制连接箭头显示点序
            for i in range(4):
                j = (i + 1) % 4
                self.draw_arrow(self.src_points[i], self.src_points[j], (0, 255, 0), 1, tip_length=0.2)

            # 在中心显示世界范围信息
            center_x = sum(p[0] for p in self.src_points) // 4
            center_y = sum(p[1] for p in self.src_points) // 4
            info_text = f"World: {self.world_width:.1f}m x {self.world_height:.1f}m"
            cv2.putText(self.canvas, info_text, (center_x - 100, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示提示信息
        if self.lane_calib_mode:
            mode_text = f"Lane Mode ({len(self.lane_lines)} lanes, {len(self.lane_temp_points) if hasattr(self, 'lane_temp_points') else 0} temp points)"
            cv2.putText(self.canvas, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            mode_text = "Perspective Mode" if self.active_mode == "src" else "Count Line Mode"
            cv2.putText(self.canvas, f"Mode: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        bev_status = "BEV Live: ON" if self.show_bev_live else "BEV Live: OFF"
        cv2.putText(self.canvas, bev_status, (self.w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.show_bev_live else (0, 0, 255), 1)

        # 显示鼠标当前位置的像素坐标（固定右上角）
        mouse_pos_text = f"Cursor: ({self.mouse_pos[0]}, {self.mouse_pos[1]})"
        cv2.putText(self.canvas, mouse_pos_text, (self.w - 330, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 显示鼠标悬停点的坐标
        if self.hover_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode == "src" else self.line_points
            if self.hover_idx < len(points):
                pt = points[self.hover_idx]
                if self.active_mode == "src" and self.hover_idx < len(self.world_points):
                    world = self.world_points[self.hover_idx]
                    coord_text = f"P{self.hover_idx + 1}: img({pt[0]},{pt[1]}) world({world[0]:.1f},{world[1]:.1f})m"
                else:
                    coord_text = f"L{self.hover_idx + 1}: ({pt[0]},{pt[1]})"
                cv2.putText(self.canvas, coord_text, (self.mouse_pos[0] + 10, self.mouse_pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(self.canvas, coord_text, (self.mouse_pos[0] + 9, self.mouse_pos[1] - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 拖动更新点位置
        if self.dragging_idx >= 0 and not self.lane_calib_mode:
            points = self.src_points if self.active_mode == "src" else self.line_points
            if self.dragging_idx < len(points):
                x, y = self.mouse_pos
                x = max(0, min(self.w, x))
                y = max(0, min(self.h, y))
                points[self.dragging_idx] = (x, y)
                if self.active_mode == "src" and len(self.src_points) == 4:
                    self.update_default_world_points()
                if self.show_bev_live and self.active_mode == "src" and len(self.src_points) == 4:
                    self.update_bev_live_preview()

    def set_road_preset(self):
        """快速设置为道路场景常用参数"""
        self.world_width = 3.5
        self.world_height = 20.0
        self.update_default_world_points()
        print("✅ 已设置为道路标准参数（宽3.5m x 长20m）")
        if self.show_bev_live:
            self.update_bev_live_preview()

    def update_default_world_points(self):
        """根据世界范围更新四个点的世界坐标（矩形）"""
        if len(self.src_points) == 4:
            self.world_points = [
                (0, 0),
                (self.world_width, 0),
                (self.world_width, self.world_height),
                (0, self.world_height)
            ]

    def draw_points(self, points, color, point_type, alpha=1.0):
        """绘制点集"""
        for i, pt in enumerate(points):
            cv2.circle(self.canvas, pt, 6, color, -1)
            cv2.circle(self.canvas, pt, 8, color, 2)
            cv2.putText(self.canvas, str(i + 1), (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.hover_idx == i and not self.lane_calib_mode and \
                    ((self.active_mode == "src" and point_type == "src") or
                     (self.active_mode == "line" and point_type == "line")):
                cv2.circle(self.canvas, pt, 12, (255, 255, 0), 2)

    def draw_arrow(self, start, end, color, thickness=2, tip_length=0.3):
        """绘制箭头"""
        start_pt = np.array(start)
        end_pt = np.array(end)
        direction = end_pt - start_pt
        length = np.linalg.norm(direction)
        if length < 1:
            return
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])
        arrow_size = min(20, length * tip_length)
        tip_point = end_pt - direction * arrow_size
        left_point = tip_point + perpendicular * (arrow_size * 0.5)
        right_point = tip_point - perpendicular * (arrow_size * 0.5)
        pts = np.array([end_pt, left_point, tip_point, right_point], dtype=np.int32)
        cv2.fillPoly(self.canvas, [pts], color)

    def set_world_range(self):
        """设置世界坐标范围（修复对话框不弹出问题）"""
        try:
            import tkinter as tk
            from tkinter import simpledialog

            # 创建根窗口并隐藏
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)  # 确保在最前

            # 使用 simpledialog 独立弹窗，不依赖 root 显示
            new_width = simpledialog.askfloat("世界宽度", "请输入宽度（米）",
                                              initialvalue=self.world_width,
                                              parent=root)
            if new_width is not None:
                self.world_width = new_width

            new_height = simpledialog.askfloat("世界高度", "请输入长度（米）",
                                               initialvalue=self.world_height,
                                               parent=root)
            if new_height is not None:
                self.world_height = new_height

            root.destroy()
            self.update_default_world_points()
            print(f"✅ 已更新世界范围: 宽={self.world_width:.1f}m, 长={self.world_height:.1f}m")

        except Exception as e:
            print(f"UI 对话框失败: {e}, 改用控制台输入")
            # 回退到控制台（见下方）
            self._set_world_range_console()

    def _set_world_range_console(self):
        """控制台备用输入"""
        print(f"\n当前世界范围: 宽={self.world_width:.1f}m, 长={self.world_height:.1f}m")
        try:
            w_input = input(f"请输入宽度（米）[默认{self.world_width:.1f}]: ").strip()
            if w_input:
                self.world_width = float(w_input)
            h_input = input(f"请输入长度（米）[默认{self.world_height:.1f}]: ").strip()
            if h_input:
                self.world_height = float(h_input)
            self.update_default_world_points()
            print(f"✅ 已更新世界范围: {self.world_width:.1f}m x {self.world_height:.1f}m")
        except ValueError:
            print("❌ 输入无效，保持原值")

    def edit_point_world_coord(self, idx):
        """编辑单个点的世界坐标"""
        if idx >= len(self.world_points):
            return

        current = self.world_points[idx]
        print(f"\n编辑点 {idx + 1} 的世界坐标（当前: x={current[0]:.2f}m, y={current[1]:.2f}m）")

        try:
            x_input = input(f"请输入新的X坐标（米）[默认{current[0]:.2f}]: ").strip()
            y_input = input(f"请输入新的Y坐标（米）[默认{current[1]:.2f}]: ").strip()

            new_x = float(x_input) if x_input else current[0]
            new_y = float(y_input) if y_input else current[1]

            self.world_points[idx] = (new_x, new_y)
            print(f"✅ 已更新点{idx + 1}世界坐标为: ({new_x:.2f}, {new_y:.2f})")

            if len(self.world_points) == 4:
                xs = [p[0] for p in self.world_points]
                ys = [p[1] for p in self.world_points]
                self.world_width = max(xs) - min(xs)
                self.world_height = max(ys) - min(ys)
                print(f"📐 世界范围已更新为: {self.world_width:.2f}m x {self.world_height:.2f}m")

            if self.show_bev_live:
                self.update_bev_live_preview()

        except ValueError:
            print("❌ 输入无效，未更新")

    def reset_points(self):
        """重置所有点"""
        self.src_points = []
        self.line_points = []
        self.lane_lines = []
        if hasattr(self, 'lane_temp_points'):
            self.lane_temp_points = []
        self.world_points = []

        margin = 50
        self.src_points = [
            (margin, margin),
            (self.w - margin, margin),
            (self.w - margin, self.h - margin),
            (margin, self.h - margin)
        ]
        self.update_default_world_points()
        print("✅ 已重置所有点")
        if self.show_bev_live:
            self.update_bev_live_preview()

    def polygon_area(self, points):
        """计算多边形有向面积"""
        area = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2

    def add_reference_grid(self, image, world_width, world_height, scale):
        """添加参考网格线"""
        img = image.copy()
        h, w = img.shape[:2]

        grid_spacing = max(1, int(min(world_width, world_height) / 5))
        grid_pixels = max(10, int(grid_spacing * scale))

        for x in range(0, w, grid_pixels):
            cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
            if x + 5 < w:
                cv2.putText(img, f"{int(x / scale)}m", (x + 2, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for y in range(0, h, grid_pixels):
            cv2.line(img, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)
            if y + 10 < h:
                cv2.putText(img, f"{int(y / scale)}m", (2, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return img

    def preview_bird_eye_view(self):
        """预览俯视图变换效果"""
        if len(self.src_points) != 4:
            print("❌ 需要4个透视点才能预览俯视图")
            return

        try:
            src = np.array(self.src_points, dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)

            src_area = self.polygon_area(src)
            dst_area = self.polygon_area(dst)

            if src_area * dst_area < 0:
                print("⚠️ 警告：源点和目标点的顺序不一致，正在自动修正...")
                dst = dst[::-1]
                self.world_points = [(p[0], p[1]) for p in dst]

            M = cv2.getPerspectiveTransform(src, dst)

            world_width = max(dst[:, 0]) - min(dst[:, 0])
            world_height = max(dst[:, 1]) - min(dst[:, 1])

            target_display_size = 800
            if world_width > 0:
                scale = target_display_size / max(world_width, world_height)
            else:
                scale = 20

            scale = max(10, min(100, scale))

            dst_width = max(100, int(world_width * scale)) if world_width > 0 else 800
            dst_height = max(100, int(world_height * scale)) if world_height > 0 else 600

            bird_view = cv2.warpPerspective(self.frame, M, (dst_width, dst_height))
            bird_view = self.add_reference_grid(bird_view, world_width, world_height, scale)

            cv2.putText(bird_view, f"BEV Preview (Scale: {scale:.1f}px/m)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bird_view, f"Region: {world_width:.2f}m x {world_height:.2f}m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 在BEV图上绘制车道线
            if self.lane_lines:
                M_inv = cv2.getPerspectiveTransform(dst, src)
                for lane_line in self.lane_lines:
                    x1, y1, x2, y2 = lane_line
                    # 将图像上的车道线转换到BEV视图
                    pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
                    world_pts = cv2.perspectiveTransform(pts, M)
                    if len(world_pts) >= 2:
                        pt1 = (int(world_pts[0][0][0]), int(world_pts[0][0][1]))
                        pt2 = (int(world_pts[1][0][0]), int(world_pts[1][0][1]))
                        # 调整到BEV图像坐标
                        scale_x = dst_width / world_width if world_width > 0 else 1
                        scale_y = dst_height / world_height if world_height > 0 else 1
                        pt1_bev = (int(pt1[0] * scale_x), int(pt1[1] * scale_y))
                        pt2_bev = (int(pt2[0] * scale_x), int(pt2[1] * scale_y))
                        cv2.line(bird_view, pt1_bev, pt2_bev, (0, 255, 255), 2)

            preview_name = "Bird's Eye View Preview (Press any key to close)"
            cv2.imshow(preview_name, bird_view)
            print("\n✅ BEV图已生成，按任意键关闭预览")
            cv2.waitKey(0)

            try:
                cv2.destroyWindow(preview_name)
            except:
                pass

        except cv2.error as e:
            print(f"❌ OpenCV透视变换失败: {e}")
        except Exception as e:
            print(f"❌ 预览失败: {e}")

    def toggle_bev_live_preview(self):
        """切换实时BEV预览模式"""
        if not self.show_bev_live:
            if len(self.src_points) == 4:
                self.show_bev_live = True
                self.create_bev_preview_window()
                print("✅ 实时BEV预览已开启")
            else:
                print("❌ 需要先选择4个透视点才能开启实时预览")
        else:
            self.close_bev_preview()
            print("✅ 实时BEV预览已关闭")

    def create_bev_preview_window(self):
        """创建实时BEV预览窗口"""
        self.bev_window = "BEV Live Preview (Auto-update)"
        cv2.namedWindow(self.bev_window)
        self.update_bev_live_preview()

    def update_bev_live_preview(self):
        """更新实时BEV预览"""
        if not self.show_bev_live or self.bev_window is None:
            return

        if len(self.src_points) != 4:
            return

        try:
            src = np.array(self.src_points, dtype=np.float32)
            dst = np.array(self.world_points, dtype=np.float32)

            src_area = self.polygon_area(src)
            dst_area = self.polygon_area(dst)
            if src_area * dst_area < 0:
                dst = dst[::-1]

            M = cv2.getPerspectiveTransform(src, dst)

            world_width = max(dst[:, 0]) - min(dst[:, 0])
            world_height = max(dst[:, 1]) - min(dst[:, 1])

            target_size = 600
            if world_width > 0:
                scale = target_size / max(world_width, world_height)
            else:
                scale = 20
            scale = max(10, min(100, scale))

            dst_width = max(100, int(world_width * scale)) if world_width > 0 else 600
            dst_height = max(100, int(world_height * scale)) if world_height > 0 else 400

            bird_view = cv2.warpPerspective(self.frame, M, (dst_width, dst_height))
            bird_view = self.add_reference_grid(bird_view, world_width, world_height, scale)

            cv2.putText(bird_view, f"Scale: {scale:.1f}px/m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(self.bev_window, bird_view)
            cv2.waitKey(1)

        except Exception:
            pass

    def close_bev_preview(self):
        """关闭实时BEV预览窗口"""
        if self.bev_window is not None:
            self.show_bev_live = False
            try:
                cv2.destroyWindow(self.bev_window)
            except:
                pass
            self.bev_window = None