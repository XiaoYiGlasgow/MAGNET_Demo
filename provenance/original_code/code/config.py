import os

# === NuPlan Dataset Configuration ===
DATA_PATH = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_PATH = '/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# 设置为绝对路径 - 使用新的按类型处理的数据
SAVE_PATH = os.path.join(BASE_DIR, "processed_data")
TRAIN_PATH = os.path.join(BASE_DIR, "processed_data_by_type", "processed_medium_magnitude_speed")
TEST_PATH = os.path.join(BASE_DIR, "processed_data_by_type", "processed_medium_magnitude_speed")
MAP_VERSION = "nuplan-maps-v1.0"

SCENARIOS_PER_TYPE = 1000000
TOTAL_SCENARIOS = 20000000

# === Temporal Configuration ===
DISCRETE_SIZE = 10  # 帧率：1秒10帧
PAST_TIME_HORIZON = 4   # 改为4秒历史
FUTURE_TIME_HORIZON = 6 # 6秒未来

NUM_PAST_POSES = DISCRETE_SIZE * PAST_TIME_HORIZON      # 40帧
NUM_FUTURE_POSES = DISCRETE_SIZE * FUTURE_TIME_HORIZON  # 60帧

# === Agent / Lane / Map Configuration ===
NUM_AGENTS = 15

LANE_NUM = 30
ROUTE_LANES_NUM = 10
CROSSWALKS_NUM = 5

LANE_POINTS_NUM = 31
ROUTE_LANES_POINTS_NUM = 51
CROSSWALKS_POINTS_NUM = 31

NODE_SIZE = (
    CROSSWALKS_NUM * (CROSSWALKS_POINTS_NUM - 1)
    + LANE_NUM * (LANE_POINTS_NUM - 1)
    + ROUTE_LANES_NUM * (ROUTE_LANES_POINTS_NUM - 1)
    + (NUM_AGENTS + 1) * NUM_PAST_POSES  # agents + ego
)

NUM_GRAPH = (
    LANE_NUM + ROUTE_LANES_NUM + CROSSWALKS_NUM + NUM_AGENTS + 1
)

QUERY_RADIUS = 30  # 轨迹提取半径

# === Training Configuration ===
WEIGHT_PATH = os.path.join(BASE_DIR, 'model_weights')

SEED = 42
EPOCHS = 100
BATCH_SIZE = 8

LEARNING_RATE = 1e-3
IN_CHANNELS = 9  # 使用9维特征
OUT_CHANNELS = NUM_FUTURE_POSES * 2  # 每个时间步预测 x, y 坐标

DECAY_LR_EVERY = 10
DECAY_LR_FACTOR = 0.5

SHOW_EVERY = 100
VAL_EVERY = 5

VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# === Transformer Architecture Hyperparameters ===
POLYLINE_FEATURE_DIM = IN_CHANNELS  # 输入特征维度
EMBED_DIM = 128                     # 嵌入维度
NUM_HEADS = 8                       # 多头注意力头数
FF_DIM = 256                        # 前馈网络隐藏层维度
NUM_LAYERS = 3                      # Transformer编码层数
OUTPUT_DIM = OUT_CHANNELS          # 输出维度

HIDDEN_DIM = 64
