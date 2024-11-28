import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver

# 航班数据
A_flights = {
    'A1': {'type': '中型', 'ETA': 100},
    'A2': {'type': '大型', 'ETA': 140},
    'A3': {'type': '中型', 'ETA': 320},
    'A4': {'type': '重型', 'ETA': 400}
}

D_flights = {
    'D1': {'type': '中型', 'ETD': 130},
    'D2': {'type': '中型', 'ETD': 180},
    'D3': {'type': '轻型', 'ETD': 270},
    'D4': {'type': '中型', 'ETD': 360},
    'D5': {'type': '轻型', 'ETD': 420},
    'D6': {'type': '大型', 'ETD': 560},
    'D7': {'type': '中型', 'ETD': 630},
    'D8': {'type': '中型', 'ETD': 700},
    'D9': {'type': '轻型', 'ETD': 200},
    'D10': {'type': '重型', 'ETD': 800}
}
wake_interval = [
    # 前序进场航班-后序进场航班
    [
        [87, 76, 76, 69],  # 轻型
        [145, 101, 76, 69],  # 中型
        [145, 101, 101, 103],  # 大型
        [174, 127, 127, 103],  # 重型
    ],
    # 前序进场航班-后序离场航班
    [
        [70, 70, 70, 70],  # 轻型
        [70, 70, 70, 70],  # 中型
        [70, 70, 70, 70],  # 大型
        [70, 70, 70, 70],  # 重型
    ],
    # 前序离场航班-后序进场航班
    [
        [112, 99, 99, 99],  # 轻型
        [112, 99, 99, 99],  # 中型
        [112, 99, 99, 99],  # 大型
        [112, 99, 99, 99],  # 重型
    ],
    # 前序离场航班-后离进场航班
    [
        [60, 60, 60, 60],  # 轻型
        [60, 60, 60, 60],  # 中型
        [60, 60, 60, 60],  # 大型
        [60, 60, 60, 60],  # 重型
    ]
]
# 时间窗口延迟限制
max_delay_a = 30  # 进场航班最大延误
max_delay_d = 30  # 离场航班最大延误

# 航班数量
num_flights = len(A_flights) + len(D_flights)

# 合并进场航班和离场航班
flights = list(A_flights.values()) + list(D_flights.values())

# 定义搜索空间
# 位置分配（每个航班一个位置）是整数空间
position_space = [Integer(0, num_flights - 1, name=f'position_{i}') for i in range(num_flights)]

# 为每个航班创建时间空间
time_space = [
    Real(min(flight['ETA'] if 'ETA' in flight else float('inf'),
             flight['ETD'] if 'ETD' in flight else float('inf')) - 10,
         max(flight['ETA'] if 'ETA' in flight else float('-inf'),
             flight['ETD'] if 'ETD' in flight else float('-inf')) + 10,
         name=f'time_{i}')
    for i, flight in enumerate(flights)
]

# 创建搜索空间
space = position_space + time_space


# 适应度函数，用来计算目标和约束
# @use_named_args(space)
def objective_function(params):
    positions = params[:num_flights]  # 获取位置
    times = params[num_flights:]  # 获取时间

    delay_sum = 0
    penalty = 0  # 用于约束惩罚

    # 确保每个位置和航班唯一
    if len(set(positions)) != num_flights:
        penalty += 1000  # 如果存在重复位置，惩罚

    # 计算累计延误总时间
    for i, flight in enumerate(flights):
        if i < len(A_flights):  # 进场航班
            eta = flight['ETA']
            t = times[i]
            # 约束 (3) 进场时间不早于 ETA
            if t < eta:
                penalty += 100
            delay_sum += max(0, t - eta)  # 延误时间，超过 ETA 的部分
        else:  # 离场航班
            etd = flight['ETD']
            t = times[i]

            # 约束 (4) 离场时间不晚于 ETD + 最大延误时间
            if t > etd + max_delay_d:
                penalty += 100

            delay_sum += max(0, t - etd)  # 延误时间，超过 ETD 的部分

    # 添加尾流间隔约束 (5) 到 (8)
    for i in range(1, num_flights):  # 从第二个位置开始检查
        prev_flight = flights[positions[i-1]]
        curr_flight = flights[positions[i]]
        type_mapping = {'轻型': 0, '中型': 1, '大型': 2, '重型': 3}
        prev_type = type_mapping[prev_flight['type']]
        curr__type = type_mapping[curr_flight['type']]

        # 根据航班类型设置不同的尾流间隔
        if 'ETA' in prev_flight and 'ETA' in curr_flight:  # 进场-进场
            t_aa = wake_interval[0][prev_type][curr__type]
            if times[i] < times[i-1] + t_aa:
                penalty += 100

        elif 'ETA' in prev_flight and 'ETD' in curr_flight:  # 进场-离场
            t_ad = wake_interval[1][prev_type][curr__type]
            if times[i] < times[i-1] + t_ad:
                penalty += 100

        elif 'ETD' in prev_flight and 'ETA' in curr_flight:  # 离场-进场
            t_da = wake_interval[2][prev_type][curr__type]
            if times[i] < times[i-1] + t_da:
                penalty += 100

        elif 'ETD' in prev_flight and 'ETD' in curr_flight:  # 离场-离场
            t_dd = wake_interval[3][prev_type][curr__type]
            if times[i] < times[i-1] + t_dd:
                penalty += 100

    return delay_sum + penalty  # 目标是最小化延误时间 + 约束惩罚



# 设置优化器
optimizer = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", n_initial_points=10, random_state=42)

# 运行优化
for i in range(100):  # 最大迭代次数为100
    # 获取下一个候选解
    x_next = optimizer.ask()

    # 评估候选解
    y_next = objective_function(x_next)

    # 提供评估结果
    optimizer.tell(x_next, y_next)
    print(f"Iteration {i + 1}, Best result: {optimizer.Xi[0]}, Best objective: {optimizer.yi[0]}")
