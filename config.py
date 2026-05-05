# Configuration File

# 输入视频目录（原始视频存放位置）
INPUT_DIR = "F:\\数据提取\\20260305\\筛选视频\\规则视频\\视频\\新沈山"
# 输出带标记的视频目录（例如绘制了检测框、关键点后的视频）
OUTPUT_VIDEO_DIR = "F:\\数据提取\\20260305\\筛选视频\\规则视频\\输出"
# 输出特征文件目录（如特征向量、JSON 标注文件等）
OUTPUT_FEATURE_DIR = "F:\\数据提取\\20260305\\筛选视频\\规则视频\\特征\\新沈山"
# calibration 相关配置
CALIB_DIR = "F:\\数据提取\\20260305\\筛选视频\\规则视频\\calibration"          # 标定文件（全局和视频专属）存放目录

# YOLO 模型路径
YOLO_MODEL_PATH = "./models/yolov8n.pt"

# 支持的视频格式
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# 透视变换标定参数 —— 图像上的四个源点（像素坐标）
SOURCE_POINTS = [(450, 300), (860, 300), (900, 600), (4, 600)]
# 世界坐标系中的实际尺寸（单位：米）
WIDTH, HEIGHT = 26.6, 107          # 宽（X方向）、长（Y方向）
# 目标点（世界坐标系中的四个角点，与 SOURCE_POINTS 一一对应）
TARGET_POINTS = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]

# （备选标定参数，已注释）
# SOURCE_POINTS = [[450, 300], [860, 300], [1900, 720], [-660, 720]]
# WIDTH, HEIGHT = 25, 100
# TARGET_POINTS = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]


# 计数线区域 —— 用于判断车辆是否通过（像素单位）
OFFSET = 55      # 计数线的宽度偏移量（左右扩展范围）
LINE_Y = 10      # 计数线的 Y 坐标（水平线的像素行数）

# 显示窗口名称
WINDOW_NAME = "Detection + Tracking + Counting + Speed Estimation"


# 保存特征字段（CSV 文件列名）
COLUMNS_FULL = [
    'Frame', 'Time(s)', 'Vehicle_ID', 'Vehicle_Type',
    'Center_X(px)', 'Center_Y(px)',
    'World_X(m)', 'World_Y(m)',
    'Length(m)', 'Width(m)',
    'Speed(km/h)', 'Acceleration(m/s²)',
    'Front_Car_ID', 'Rear_Car_ID',
    'Left_Front_Car_ID', 'Left_Rear_Car_ID',
    'Right_Front_Car_ID', 'Right_Rear_Car_ID',
    'Lane_ID',
     # 'Calib_Version'
]

# 自动选择模式（True 时跳过交互式选点，使用预设标定参数）
AUTO_MODE = False

# ROI（感兴趣区域）相关配置
USE_ROI = True               # 是否启用 ROI 区域限制
ROI_DIR = "calibration"      # ROI 文件保存目录

# 缩放检测与自动重标定参数（用于应对镜头变焦或移动）
STABLE_NEEDED = 10           # 需要连续多少帧稳定后触发重标定
ZOOM_THRESH = 0.05           # 缩放变化阈值（超过该值认为发生缩放）

AUTO_ZOOM_DETECTION = True   # 是否启用自动缩放检测（设为 False 则关闭）

# 批量处理时是否跳过已提取特征的视频
SKIP_EXISTING_FEATURES = True

# ------------------ 卡尔曼滤波参数 ------------------
# 每个目标轨迹的滑动窗口最大时长（单位：秒）
MAX_HISTORY_SECONDS = 1
# 过程噪声协方差 Q：表示速度自然波动的程度，值越小越信任匀速假设
KALMAN_Q = 0.1
# 测量噪声协方差 R：表示原始速度测量值的噪声大小，值越小越信任观测值
KALMAN_R = 5.0
# 初始估计误差协方差 P：通常取较大值，使滤波器快速收敛
KALMAN_P_INIT = 10.0