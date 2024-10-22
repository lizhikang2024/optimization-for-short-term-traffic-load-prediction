import numpy as np
import random
import math
# start
import time
import torch
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# end

global_gru_train_and_pred_count = 1
global_fitness_data = 100000
global_dbo_window = 0
global_dbo_hidden = 0
global_dbo_lr = 0
global_dbo_dropout = 0


class funtion():
    def __init__(self):
        print("starting DBO")


def Parameters(F):
    if F == 'F1':
        # ParaValue=[-100,100,30] [-100,100]代表初始范围，30代表dim维度
        fobj = F1
        lb = -100
        ub = 100
        dim = 3
    elif F == 'F2':
        fobj = F2
        lb = -10
        ub = 10
        dim = 30
    elif F == 'F3':
        fobj = F3
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F4':
        fobj = F4
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F5':
        fobj = F5
        lb = -30
        ub = 30
        dim = 30
    elif F == 'F6':
        fobj = F6
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F7':
        fobj = F7
        lb = -1.28
        ub = 1.28
        dim = 30
    elif F == 'F8':
        fobj = F8
        lb = -500
        ub = 500
        dim = 30
    elif F == 'F9':
        fobj = F9
        lb = -5.12
        ub = 5.12
        dim = 30

    elif F == 'F10':
        fobj = F10
        lb = -32
        ub = 32
        dim = 30

    elif F == 'F11':
        fobj = F11
        lb = -600
        ub = 600
        dim = 30

    elif F == 'F12':
        fobj = F12
        lb = -50
        ub = 50
        dim = 30

    elif F == 'F13':
        fobj = F13
        lb = -50
        ub = 50
        dim = 30

    elif F == 'F14':
        fobj = F14
        lb = -65.536
        ub = 65.536
        dim = 2

    elif F == 'F15':
        fobj = F15
        lb = -5
        ub = 5
        dim = 4

    elif F == 'F16':
        fobj = F16
        lb = -5
        ub = 5
        dim = 2

    elif F == 'F17':
        fobj = F17
        lb = [-5, 0]
        ub = [10, 15]
        dim = 2

    elif F == 'F18':
        fobj = F18
        lb = -2
        ub = 2
        dim = 2

    elif F == 'F19':
        fobj = F19
        lb = 0
        ub = 1
        dim = 3

    elif F == 'F20':
        fobj = F20
        lb = 0
        ub = 1
        dim = 6

    elif F == 'F21':
        fobj = F21
        lb = 0
        ub = 10
        dim = 4

    elif F == 'F22':
        fobj = F22
        lb = 0
        ub = 10
        dim = 4

    elif F == 'F23':
        fobj = F23
        lb = 0
        ub = 10
        dim = 4

    elif F == 'F24':
        fobj = F24
        lb = 0
        ub = 10
        dim = 4
    return fobj, lb, ub, dim


# start
# 随机数种子
np.random.seed(0)


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def calculate_rmse(y_true, y_pred):
    # 均方根误差
    rmse = math.sqrt(np.mean(np.square(y_true - y_pred)))
    return rmse


def create_inout_sequences(input_data, tw, pre_len):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + 4) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# end

"""
F24适应度函数为GRU网络超参数优化特别设计
"""


def F24(x):
    global global_gru_train_and_pred_count
    dataset_choice = 1
    # epoch数
    change_epoch = 5
    """
    DBO将优化的GRU超参数
    """
    dbo_window = int(12 + 1.5 * x[0])
    dbo_hidden = int(30 + 5 * x[1])
    dbo_lr = round(0.001 * x[2], 4)
    dbo_dropout = round(0.04 * x[3], 3)

    print("****************************************")
    gru_inf = '本次为全局第 ' + str(global_gru_train_and_pred_count) + ' 次GRU模型训练与预测，超参数为：'
    print(gru_inf)
    print("观测窗口大小：", dbo_window)
    print("隐藏层神经元数：", dbo_hidden)
    print("学习率：", dbo_lr)
    print("dropout参数：", dbo_dropout)
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print("****************************************", file=plog_write)
    print('本次DBO参数变量为：', x, file=plog_write)
    gru_inf = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ' ------ ' + '本次为全局第 ' + str(
        global_gru_train_and_pred_count) + ' 次GRU模型训练与预测，超参数为：'
    print(gru_inf, file=plog_write)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 观测窗口大小：", dbo_window,
          file=plog_write)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 隐藏层神经元数：", dbo_hidden,
          file=plog_write)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 学习率：", dbo_lr, file=plog_write)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ dropout参数：", dbo_dropout,
          file=plog_write)
    plog_write.close()
    """
    GRU模型定义部分
    """

    class GRU(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=dbo_hidden, num_layers=2, output_dim=1, pre_len=1):
            super(GRU, self).__init__()
            self.pre_len = pre_len
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            # 替换 LSTM 为 GRU
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dbo_dropout)

        def forward(self, x):
            h0_gru = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

            out, _ = self.gru(x, h0_gru)

            out = self.dropout(out)

            # 取最后 pre_len 时间步的输出
            out = out[:, -self.pre_len:, :]

            out = self.fc(out)
            out = self.relu(out)
            return out

    """
    数据定义部分
    """
    if dataset_choice == 0:
        true_data = pd.read_csv('ETTh1.csv', nrows=8000)  # 填你自己的数据地址,自动选取你最后一列数据为特征列
        # true_data = pd.read_csv('ETTh1.csv')  # 填你自己的数据地址,自动选取你最后一列数据为特征列
        target = 'OT'  # 添加你想要预测的特征列
    else:
        true_data = pd.read_csv('ukdata.csv', nrows=8000)  # 填你自己的数据地址,自动选取你最后一列数据为特征列
        # true_data = pd.read_csv('ETTh1.csv')  # 填你自己的数据地址,自动选取你最后一列数据为特征列
        target = 'data'  # 添加你想要预测的特征列

    test_size = 0.2  # 训练集和测试集的尺寸划分
    train_size = 0.8  # 训练集和测试集的尺寸划分
    pre_len = 1  # 预测未来数据的长度
    train_window = dbo_window  # 观测窗口

    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = np.array(true_data[target])

    # 定义标准化优化器X
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))

    # 训练集和测试集划分
    train_data = true_data[:int(train_size * len(true_data))]
    test_data = true_data[-int(test_size * len(true_data)):]
    print("训练集尺寸:", len(train_data))
    print("测试集尺寸:", len(test_data))
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print("训练集尺寸:", len(train_data), file=plog_write)
    print("测试集尺寸:", len(test_data), file=plog_write)
    plog_write.close()

    # 进行标准化处理
    train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
    test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized)
    test_data_normalized = torch.FloatTensor(test_data_normalized)

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)

    # 创建 DataLoader
    batch_size = dbo_window  # 你可以根据需要调整批量大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 模型部分参数准备
    gru_model = GRU(input_dim=1, output_dim=1, num_layers=2, hidden_dim=dbo_hidden, pre_len=pre_len)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=dbo_lr)
    epochs = change_epoch

    # 根据相关超参数进行模型训练
    losss = []

    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 开始训练GRU模型!", file=plog_write)
    plog_write.close()

    gru_model.train()  # 训练模式
    start_time = time.time()  # 计算起始时间
    for i in range(epochs):
        for seq, labels in train_loader:
            gru_model.train()
            optimizer.zero_grad()
            y_pred = gru_model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            # print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        losss.append(single_loss.detach().numpy())
    filename = './data_of_dbo-gru/result_model/save_model_gru_' + str(global_gru_train_and_pred_count) + '.pth'
    torch.save(gru_model.state_dict(), filename)

    print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          f"------ 模型已保存,用时:{(time.time() - start_time) / 60:.4f} min", file=plog_write)
    plog_write.close()

    # 根据训练结果模型进行预测
    gru_model.load_state_dict(torch.load(filename))
    gru_model.eval()  # 评估模式
    results = []
    reals = []
    losss_mae = []
    losss_rmse = []

    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 利用GRU模型开始预测!",
          file=plog_write)
    plog_write.close()

    for seq, labels in test_loader:
        pred = gru_model(seq)
        mae = calculate_mae(pred.detach().numpy(), np.array(labels))  # MAE误差计算绝对值(预测值  - 真实值)
        rmse = calculate_rmse(pred.detach().numpy(), np.array(labels))  # MAE误差计算绝对值(预测值  - 真实值)
        losss_mae.append(mae)
        losss_rmse.append(rmse)
        for j in range(batch_size):
            for i in range(pre_len):
                reals.append(labels[j][i][0].detach().numpy())
                results.append(pred[j][i][0].detach().numpy())

    reals = scaler_test.inverse_transform(np.array(reals).reshape(1, -1))[0]
    results = scaler_test.inverse_transform(np.array(results).reshape(1, -1))[0]

    csv_filename = './data_of_dbo-gru/result_csv/real_and_pre_info_dbo-gru_' + str(
        global_gru_train_and_pred_count) + '.csv'
    real_and_pre_info = pd.DataFrame({'reals': reals, 'predict': results})
    real_and_pre_info.to_csv(csv_filename, index=False)

    print("****************************************")
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 模型预测结果：", results,
    #       file=plog_write)
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 预测误差MAE:", losss_mae,
    #       file=plog_write)
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 预测误差RMSE:", losss_rmse,
    #       file=plog_write)
    print("****************************************", file=plog_write)
    plog_write.close()

    ave_mae = np.mean(losss_mae)
    ave_rmse = np.mean(losss_rmse)

    # 画图保存直观的预测结果
    plt.figure()
    plt.style.use('ggplot')
    # 创建折线图
    plt.plot(reals, label='real', color='blue')  # 实际值
    plt.plot(results, label='forecast', color='red', linestyle='--')  # 预测值
    # 增强视觉效果
    plt.grid(True)
    plt.title('real vs forecast')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    filename = './data_of_dbo-gru/result_pic/test_results_' + str(global_gru_train_and_pred_count) + '.png'
    plt.savefig(filename)
    plt.close('all')

    global_gru_train_and_pred_count += 1

    o = ave_mae + ave_rmse
    print('上一GRU模型训练预测后，获取DBO适应度为：', o)

    global global_fitness_data
    global global_dbo_window
    global global_dbo_hidden
    global global_dbo_lr
    global global_dbo_dropout
    if o <= global_fitness_data:
        global_fitness_data = o
        global_dbo_window = dbo_window
        global_dbo_hidden = dbo_hidden
        global_dbo_lr = dbo_lr
        global_dbo_dropout = dbo_dropout
    print('截至此刻，最佳参数为：\n', 'dbo_window =', global_dbo_window, 'dbo_hidden =', global_dbo_hidden, 'dbo_lr =',
          global_dbo_lr, 'dbo_dropout =', global_dbo_dropout)

    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          '------ 上一GRU模型训练预测后，获取DBO适应度为：', o, file=plog_write)
    print('截至此刻，最佳参数为：\n', 'dbo_window =', global_dbo_window, 'dbo_hidden =', global_dbo_hidden, 'dbo_lr =',
          global_dbo_lr, 'dbo_dropout =', global_dbo_dropout, file=plog_write)
    plog_write.close()
    return o


# F1

def F1(x):
    o = np.sum(np.square(x))
    return o


# F2
def F2(x):
    o = np.sum(np.abs(x)) + np.prod(np.abs(x))
    return o


# F3
def F3(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o = o + np.square(np.sum(x[0:i]))
    return o


# F4
def F4(x):
    o = np.max(np.abs(x))
    return o


# F5
def F5(x):
    dim = len(x)
    o = np.sum(100 * np.square(x[1:dim] - np.square(x[0:dim - 1]))) + np.sum(np.square(x[0:dim - 1] - 1))
    return o


# F6
def F6(x):
    o = np.sum(np.square(np.abs(x + 0.5)))
    return o


# F7
def F7(x):
    dim = len(x)
    num1 = [num for num in range(1, dim + 1)]
    o = np.sum(num1 * np.power(x, 4)) + np.random.rand(1)
    return o


# F8
def F8(x):
    o = np.sum(0 - x * np.sin(np.sqrt(np.abs(x))))
    return o


# F9
def F9(x):
    dim = len(x)
    o = np.sum(np.square(x) - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return o


# F10
def F10(x):
    dim = len(x)
    o = 0 - 20 * np.exp(0 - 0.2 * np.sqrt(np.sum(np.square(x)) / dim)) - np.exp(
        np.sum(np.cos(2 * math.pi * x)) / dim) + 20 + np.exp(1)
    return o


# F11
def F11(x):
    dim = len(x)
    num1 = [num for i in range(1, dim + 1)]
    o = np.sum(np.square(x)) / 4000 - np.prod(np.cos(x / np.sqrt(num1))) + 1
    return o


# F12
def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (10 * np.square(np.sin(math.pi * (1 + (x[1] + 1) / 4))) + \
                           np.sum((((x[0:dim - 2] + 1) / 4) ** 2) * \
                                  (1 + 10. * ((np.sin(math.pi * (1 + (x[1:dim - 1] + 1) / 4)))) ** 2)) + (
                                   (x[dim - 1] + 1) / 4) ** 2) + np.sum(Ufun(x, 10, 100, 4))
    return o


# F13
def F13(x):
    dim = len(x)
    o = 0.1 * (np.square(np.sin(3 * math.pi * x[1])) + np.sum(
        np.square(x[0:dim - 2] - 1) * (1 + np.square(np.sin(3 * math.pi * x[1:dim - 1])))) + \
               np.square(x[dim - 1] - 1) * (1 + np.square(np.sin(2 * math.pi * x[dim - 1])))) + np.sum(
        Ufun(x, 5, 100, 4))

    return o


def Ufun(x, a, k, m):
    o = k * np.power(x - a, m) * (x > a) + k * (np.power(-x - a, m)) * (x < (-a))
    return o


# F14
def F14(x):
    aS = [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32], \
          [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]
    ss = x.T
    bS = np.zeros
    bS = aS[0, :]
    num1 = [num for num in range(1, 26)]
    for j in range(25):
        bS[j] = np.sum(np.power(ss - aS[:, j], 6))
    o = 1 / (1 / 500 + np.sum(1 / (num1 + bS)))
    return o


# F15
def F15(x):
    aK = np.array[0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    bK = np.array[0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    bK = 1 / bK
    o = np.sum(np.square(aK - (x[1] * (np.square(bK) + x[2] * bK)) / (np.square(bK) + x[3] * bK + x[4])))
    return o


# F16
def F16(x):
    o = 4 * np.square(x[1]) - 2.1 * np.power(x[1], 4) + np.power(x[1], 6) / 3 + \
        x[1] * x[2] - 4 * np.square(x[2]) + 4 * np.power(x[2], 4)
    return o


# F17
def F17(x):
    o = np.square(x[2] - np.square(x[1]) * 5.1 / (4 * np.square(math.pi)) + 5 / math.pi * x[1] - 6) + \
        10 * (1 - 1 / (8 * math.pi)) * np.cos(x[1]) + 10
    return o


# F18
def F18(x):
    o = (1 + np.square(x[1] + x[2] + 1) * (
            19 - 14 * x[1] + 3 * np.square(x[1]) - 14 * x[2] + 6 * x[1] * x[2] + 3 * np.square(x[2]))) * \
        (30 + np.square(2 * x[1] - 3 * x[2]) * (
                18 - 32 * x[1] + 12 * np.square(x[1]) + 48 * x[2] - 36 * x[1] * x[2] + 27 * np.square(x[2])))
    return o


# F19
def F19(x):
    aH = np.array[[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    cH = np.array[1, 1.2, 3, 3.2]
    pH = np.array[[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]]
    o = 0;
    for i in range(4):
        o = o - cH[i] * np.exp(0 - np.sum(aH[i, :] * np.square(x - pH[i, :])))
    return o


# F20
def F20(x):
    aH = np.array[
        [10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
    cH = np.array[1, 1.2, 3, 3.2]
    pH = np.array[
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348,
                                                                                                             0.1415,
                                                                                                             0.3522,
                                                                                                             0.2883,
                                                                                                             0.3047,
                                                                                                             0.6650], [
            0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
    o = 0
    for i in range(4):
        o = o - cH(i) * np.exp(0 - (np.sum(aH[i, :] * (np.square(x - pH[i, :])))))
    return o


# F21
def F21(x):
    aSH = np.array[
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8,
                                                                                                           1], [6, 2, 6,
                                                                                                                2], [7,
                                                                                                                     3.6,
                                                                                                                     7,
                                                                                                                     3.6]]
    cSH = np.array[0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    o = 0
    for i in range(5):
        o = o - 1 / (np.sum((x - aSH[i, :]) * (x - aSH[i, :])) + cSH(i))
    return o


# F22
def F22(x):
    aSH = np.array[
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8,
                                                                                                           1], [6, 2, 6,
                                                                                                                2], [7,
                                                                                                                     3.6,
                                                                                                                     7,
                                                                                                                     3.6]]
    cSH = np.array[0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    o = 0
    for i in range(7):
        o = o - 1 / (np.sum((x - aSH[i, :]) * (x - aSH[i, :])) + cSH(i))
    return o


# F23
def F23(x):
    aSH = np.array[
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8,
                                                                                                           1], [6, 2, 6,
                                                                                                                2], [7,
                                                                                                                     3.6,
                                                                                                                     7,
                                                                                                                     3.6]]
    cSH = np.array[0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    o = 0
    for i in range(10):
        o = o - 1 / (np.sum((x - aSH[i, :]) * (x - aSH[i, :])) + cSH[i])
    return o


def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp


def Boundss(ss, LLb, UUb):
    temp = ss
    for i in range(len(ss)):
        if temp[i] < LLb[0, i]:
            temp[i] = LLb[0, i]
        elif temp[i] > UUb[0, i]:
            temp[i] = UUb[0, i]
    return temp


def swapfun(ss):
    temp = ss
    o = np.zeros((1, len(temp)))
    for i in range(len(ss)):
        o[0, i] = temp[i]
    return o


def DBO(pop, M, c, d, dim, fun):
    """
    :param fun: 适应度函数
    :param pop: 种群数量
    :param M: 迭代次数
    :param c: 迭代范围下界
    :param d: 迭代范围上界
    :param dim: 优化参数的个数
    :return: 适应度值最小的值 对应得位置
    """
    P_percent = 0.2
    pNum = round(pop * P_percent)
    lb = c * np.ones((1, dim))
    ub = d * np.ones((1, dim))
    X = np.zeros((pop, dim))
    fit = np.zeros((pop, 1))

    global global_gru_train_and_pred_count
    global_gru_train_and_pred_count = 1

    print("===================================================")
    print('开始初始化种群，数量为：', pop)
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print("===================================================", file=plog_write)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '------ 开始初始化种群，数量为：', pop,
          file=plog_write)
    plog_write.close()

    start_time = time.time()
    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        fit[i, 0] = fun(X[i, :])
    pFit = fit
    pX = X
    XX = pX
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]
    Convergence_curve = np.zeros((1, M))

    print(f"种群初始化结束,用时:{(time.time() - start_time) / 60:.4f} min")
    print("===================================================")
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          f"------ 种群初始化结束,用时:{(time.time() - start_time) / 60:.4f} min", file=plog_write)
    print("===================================================", file=plog_write)
    plog_write.close()

    for t in range(M):
        print("===================================================")
        iteration_inf = '开始第 ' + str(t + 1) + ' 次迭代！'
        print(iteration_inf)
        # 记录信息到日志文件中
        plog_write = open('DBO_plog.txt', 'a')
        print("===================================================", file=plog_write)
        iteration_inf = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ' ------ 开始第 ' + str(
            t + 1) + ' 次迭代！'
        print(iteration_inf, file=plog_write)
        plog_write.close()

        start_time = time.time()
        # sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0])
        worse = X[B, :]  #
        r2 = np.random.rand(1)
        # v0 = 0.5
        # v = v0 + t/(3*M)
        for i in range(pNum):
            if r2 < 0.9:
                r1 = np.random.rand(1)
                a = np.random.rand(1)
                if a > 0.1:
                    a = 1
                else:
                    a = -1
                X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])  # Equation(1)
            else:
                aaa = np.random.randint(180, size=1)
                if aaa == 0 or aaa == 90 or aaa == 180:
                    X[i, :] = pX[i, :]
                theta = aaa * math.pi / 180
                X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])  # Equation(2)
            X[i, :] = Bounds(X[i, :], lb, ub)
            fit[i, 0] = fun(X[i, :])
        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :]

        R = 1 - t / M
        Xnew1 = bestXX * (1 - R)
        Xnew2 = bestXX * (1 + R)
        Xnew1 = Bounds(Xnew1, lb, ub)  # Equation(3)
        Xnew2 = Bounds(Xnew2, lb, ub)
        Xnew11 = bestX * (1 - R)
        Xnew22 = bestX * (1 + R)  # Equation(5)
        Xnew11 = Bounds(Xnew11, lb, ub)
        Xnew22 = Bounds(Xnew22, lb, ub)
        xLB = swapfun(Xnew1)
        xUB = swapfun(Xnew2)

        for i in range(pNum + 1, 12):  # Equation(4)
            X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                    pX[i, :] - Xnew2)
            X[i, :] = Bounds(X[i, :], xLB, xUB)
            fit[i, 0] = fun(X[i, :])
        for i in range(13, 19):  # Equation(6)
            X[i, :] = pX[i, :] + (
                    (np.random.randn(1)) * (pX[i, :] - Xnew11) + ((np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
            X[i, :] = Bounds(X[i, :], lb, ub)
            fit[i, 0] = fun(X[i, :])
        for j in range(20, pop):  # Equation(7)
            X[j, :] = bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
            X[j, :] = Bounds(X[j, :], lb, ub)
            fit[j, 0] = fun(X[j, :])

        # Update the individual's best fitness vlaue and the global best fitness value
        XX = pX
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]

        Convergence_curve[0, t] = fMin

        iteration_inf = '第 ' + str(t + 1) + ' 次迭代结束！'
        print(iteration_inf)
        print(f"此轮DBO迭代结束,用时:{(time.time() - start_time) / 60:.4f} min")
        print("===================================================")
        # 记录信息到日志文件中
        plog_write = open('DBO_plog.txt', 'a')
        iteration_inf = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ' ------ ' + '第 ' + str(
            t + 1) + ' 次迭代结束！'
        print(iteration_inf, file=plog_write)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
              f"------ 此轮DBO迭代结束,用时:{(time.time() - start_time) / 60:.4f} min", file=plog_write)
        print("===================================================", file=plog_write)
        plog_write.close()

    return fMin, bestX, Convergence_curve
