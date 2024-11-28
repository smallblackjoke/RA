import numpy as np
from scipy.optimize import differential_evolution
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import time
# 航班数据
A_flights = {
    'A1': {'id':'A1','type': '中型', 'ETA': 100},
    'A2': {'id':'A2','type': '大型', 'ETA': 140},
    'A3': {'id':'A3','type': '中型', 'ETA': 320},
    'A4': {'id':'A4','type': '重型', 'ETA': 400}
}

D_flights = {
    'D1': {'id':'D1','type': '中型', 'ETD': 130},
    'D2': {'id':'D2','type': '中型', 'ETD': 180},
    'D3': {'id':'D3','type': '轻型', 'ETD': 270},
    'D4': {'id':'D4','type': '中型', 'ETD': 360},
    'D5': {'id':'D5','type': '轻型', 'ETD': 420},
    'D6': {'id':'D6','type': '大型', 'ETD': 560},
    'D7': {'id':'D7','type': '中型', 'ETD': 630},
    'D8': {'id':'D8','type': '中型', 'ETD': 700},
    'D9': {'id':'D9','type': '轻型', 'ETD': 200},
    'D10': {'id':'D10','type': '重型', 'ETD': 800}
}

type_mapping = {'轻型': 0, '中型': 1, '大型': 2, '重型': 3}

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
max_delay_a = 500  # 进场航班最大延误
max_delay_d = 500  # 离场航班最大延误

# 航班数量
num_flights = len(A_flights) + len(D_flights)

# 合并进场航班和离场航班
flights = list(A_flights.values()) + list(D_flights.values())

# 定义搜索空间
# # 位置分配（每个航班一个位置）是整数空间
# position_space = [Integer(0, num_flights - 1, name=f'position_{i}') for i in range(num_flights)]
#
# # 为每个航班创建时间空间
# time_space = [
#     Real(min(flight['ETA'] if 'ETA' in flight else float('inf'),
#              flight['ETD'] if 'ETD' in flight else float('inf')) - 10,
#          max(flight['ETA'] if 'ETA' in flight else float('-inf'),
#              flight['ETD'] if 'ETD' in flight else float('-inf')) + 10,
#          name=f'time_{i}')
#     for i, flight in enumerate(flights)
# ]
# # 跑道分配空间（每个航班可以选择跑道 1 或 跑道 2）
# runway_space = [Integer(0, 1, name=f'runway_{i}') for i in range(num_flights)]
#
# # 创建搜索空间
# space = position_space + time_space + runway_space


# 适应度函数，用来计算目标和约束
# @use_named_args(space)
# 定义搜索空间的边界
position_bound = (0, num_flights - 1)  # 位置的范围
Del = 500  # 最大延误量
upper = 800
lower = 100
time_bound = [(flight['ETA'],min(flight['ETA']+Del,upper)) if 'ETA' in flight else (max(flight['ETD']-Del,lower),flight['ETD'])
              for flight in flights]  # 时间的范围
runway_bound = (0, 1)

bounds = []
for _ in range(num_flights):  # 位置空间
    bounds.append(position_bound)
for i in range(len(flights)):  # 时间空间
    bounds.append(time_bound[i])
for _ in range(num_flights):
    bounds.append(runway_bound)

def objective_function(params):
    positions = np.round(params[:num_flights])  # 获取位置
    times = params[num_flights:num_flights * 2]  # 获取时间
    runways = np.round(params[num_flights * 2:])  # 获取跑道分配

    delay_sum = 0
    penalty = 0  # 用于约束惩罚

    # 记录各惩罚项
    pos_conflict = 0
    time_over = 0
    runway_conflict = 0
    # 确保每个位置和航班唯一
    pos_tuple = []
    for i in range(num_flights):
        pos_tuple.append((positions[i],runways[i]))
    if len(set(pos_tuple)) != num_flights:
        penalty += 1000  # 如果存在重复位置，惩罚
        pos_conflict += 1

    # 计算累计延误总时间
    for i, flight in enumerate(flights):
        if i < len(A_flights):  # 进场航班
            eta = flight['ETA']
            t = times[i]
            # 约束 (3) 进场时间不早于 ETA,或者超过最大延误时间
            if t < eta or t> eta + 500:
                penalty += 100
                time_over+=1
            delay_sum += max(0, t - eta)  # 延误时间，超过 ETA 的部分
        else:  # 离场航班
            etd = flight['ETD']
            t = times[i]

            # 约束 (4) 离场时间不早于 ETD - 最大延误时间，或者不晚于ETD
            if t < etd - max_delay_d or t > etd:
                penalty += 100
                time_over += 1

            delay_sum += max(0, etd - t)  # 延误时间，提前 ETD 的部分

    # 添加尾流间隔约束 (5) 到 (8)
    # 创建一个字典，将每条跑道的飞机按使用时间排序
    runway_dict = {}
    for i in range(len(runways)):
        r = int(runways[i])  # 将跑道的标记转换为整数
        if r not in runway_dict:
            runway_dict[r] = []
        runway_dict[r].append((times[i], int(positions[i]),i))  # 存储时间,位置和编号
    # 对每条跑道上的飞机按照时间进行排序
    for r in runway_dict:
        runway_dict[r].sort()  # 根据时间排序
    for r in runway_dict:
        prev = runway_dict[r][0]
        for i in range(1,len(runway_dict[r])):
            curr = runway_dict[r][i]
            prev_flight = flights[prev[2]]
            curr_flight = flights[curr[2]]
            prev_type = type_mapping[prev_flight['type']]
            curr__type = type_mapping[curr_flight['type']]
            # 根据航班类型设置不同的尾流间隔
            if 'ETA' in prev_flight and 'ETA' in curr_flight:  # 进场-进场
                t_aa = wake_interval[0][prev_type][curr__type]
                if prev[0] + t_aa - curr[0] > 0:
                    penalty += 1000
                    runway_conflict+=1
            elif 'ETA' in prev_flight and 'ETD' in curr_flight:  # 进场-离场
                t_ad = wake_interval[1][prev_type][curr__type]
                if prev[0] + t_ad - curr[0] > 0:
                    penalty += 1000
                    runway_conflict += 1

            elif 'ETD' in prev_flight and 'ETA' in curr_flight:  # 离场-进场
                t_da = wake_interval[2][prev_type][curr__type]
                if prev[0] + t_da - curr[0] > 0:
                    penalty += 1000
                    runway_conflict += 1

            elif 'ETD' in prev_flight and 'ETD' in curr_flight:  # 离场-离场
                t_dd = wake_interval[3][prev_type][curr__type]
                if prev[0] + t_dd - curr[0] > 0:
                    penalty += 1000
                    runway_conflict += 1
            prev = curr

    # 第2条跑道只允许离场航班使用
    if len(runway_dict) > 1:
        for i in range(len(runway_dict[1])):
            # 前4个航班为进场航班。
            curr = runway_dict[1][i]
            if curr[2] < 4 :
                penalty += 1000

    print("位置冲突：{} 时间越界：{} 跑道间隔冲突：{}".format(pos_conflict,time_over,runway_conflict))
    return delay_sum + penalty  # 目标是最小化延误时间 + 约束惩罚


def run_differential_evolution(bounds, restart=False, init_solution=None):
    # 如果是重启优化，则使用上一次的最优解作为初始解
    if restart and init_solution is not None:
        result = differential_evolution(objective_function, bounds, maxiter=100, popsize=30, mutation=(0.5, 1),
                                        recombination=0.7, init=init_solution)
    else:
        # 初次优化或没有初始化解
        result = differential_evolution(objective_function, bounds, maxiter=100, popsize=30, mutation=(0.5, 1),
                                        recombination=0.7)

    return result

if __name__ == '__main__':
    # result = differential_evolution(objective_function, bounds, maxiter=20, popsize=30, mutation=(0.5, 1), recombination=0.7)
    for k in range(1):
        start_time = time.time()
        # result = run_differential_evolution(bounds, restart=True,init_solution=result.x)
        result = differential_evolution(objective_function, bounds, maxiter=1000, popsize=15, mutation=(0.5, 1), recombination=0.7)
        end_time = time.time()
        # 计算并输出执行时间
        elapsed_time = end_time - start_time
        print(f"代码执行时间: {elapsed_time} 秒")
        print("Best solution found: ", result.x)
        print("Best objective value: ", result.fun)
        # 保存最佳解
        np.save('best_solution.npy', result.x)

        # # 加载最佳解
        # best_solution = np.load('best_solution.npy')

        positions = np.round(result.x[:num_flights])  # 获取位置
        times = result.x[num_flights:num_flights * 2]  # 获取时间
        runways = np.round(result.x[num_flights * 2:])  # 获取跑道分配
        # 获取航班号
        flight_numbers = np.arange(1, len(times) + 1)  # 假设航班号为1到N
        # 计算总延误时间
        all_del = 0
        for i in range(num_flights):
            flight = flights[i]
            if 'ETA' in flight:
                all_del += times[i] - flight['ETA']
            else:
                all_del += flight['ETD']  - times[i]
        print("总延误时间：{}秒".format(all_del))
        # 创建图形
        plt.figure(figsize=(10, 6))

        # 在图上画出每个航班的调度点
        for i in range(len(runways)):
            plt.scatter(runways[i], times[i], label=f"flight {flight_numbers[i]}", s=100)  # 使用scatter绘制点

        # 标注每个点
        for i in range(len(runways)):
            flight = flights[i]
            describe = "{}:ET{},ST{}".format(flight['id'],flight['ETA'] if 'ETA' in flight else flight['ETD'],int(times[i]))
            plt.text(runways[i], times[i], f"{describe}", fontsize=9, ha='center', va='bottom')

        # 设置图表的标题和标签
        plt.title('runway-sort{}'.format(k))
        plt.xlabel('runway')
        plt.ylabel('time')
        # 显示网格
        plt.grid(True)
        # 设置y轴范围（时间区间）
        plt.ylim(min(times) - 50, max(times) + 50)
        # 设置x轴范围（跑道0和1）
        plt.xlim(-0.5, 1.5)
        # 显示图例
        plt.legend()
        # 展示图形
        plt.show()



# 设置优化器
# optimizer = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", n_initial_points=10, random_state=42)

# # 运行优化
# for i in range(100):  # 最大迭代次数为100
#     # 获取下一个候选解
#     x_next = optimizer.ask()
#
#     # 评估候选解
#     y_next = objective_function(x_next)
#
#     # 提供评估结果
#     optimizer.tell(x_next, y_next)
#     print(f"Iteration {i + 1}, Best result: {optimizer.Xi[0]}, Best objective: {optimizer.yi[0]}")
