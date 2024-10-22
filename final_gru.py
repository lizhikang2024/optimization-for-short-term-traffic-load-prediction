import time
import math
import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
import os
from pathlib import Path
import seaborn as sns

sns.set()

# from torch.utils.data import Datasetpip
dataset_choice = 0  # 0 表示电网电荷流量数据， 1 表示英国学术数据中心网络流量数据
tmp_path = ['./data_of_gru/etth', './data_of_gru/ukdata']


def clear_folder(folder_path):
    # 利用pathlib模块将输入的文件夹路径转换为Path对象
    folder_path = Path(folder_path)

    # 遍历文件夹内的所有文件
    for file_path in folder_path.iterdir():
        # 判断当前文件路径是否为文件夹
        if file_path.is_dir():
            # 如果是文件夹，则递归调用clear_folder函数清空文件夹内的文件
            clear_folder(file_path)
        else:
            # 如果是文件，则直接删除该文件
            file_path.unlink()


def draw_28_types_pic_with_two_datalist(data1, data2, save_path):
    style_list = ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic',
                  'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8',
                  'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette',
                  'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
                  'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk',
                  'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
    for i in range(len(style_list)):
        plt.figure()
        plt.style.use(style_list[i])

        # 创建折线图
        plt.plot(data1, label='real')
        plt.plot(data2, label='forecast', linestyle='--')

        # 增强视觉效果
        plt.grid(True)
        plt.title('real vs forecast---' + style_list[i])
        plt.xlabel('episode')
        plt.ylabel('reward value')
        # plt.legend()
        #  plt.show()
        plt.savefig(save_path + '/test_results_gru_pic_' + str(i + 1) + '.png')
        plt.close('all')


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


"""
DBO将优化的GRU超参数
"""
if dataset_choice == 0:
    dbo_window = 12
    dbo_hidden = 43
    dbo_lr = 0.0003
    dbo_dropout = 0.31
else:  # dataset_choice=1
    dbo_window = 24
    dbo_hidden = 40
    dbo_lr = 0.0011
    dbo_dropout = 0.4
# epoch数
change_epoch = 10
"""
检查etth或ukdata文件夹是否存在于data_of_tcn文件夹中，不存在则新建，存在则清空子文件夹
"""
if os.path.exists(tmp_path[dataset_choice]):
    clear_folder(tmp_path[dataset_choice])
else:
    os.makedirs(tmp_path[dataset_choice])
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

# 定义标准化优化器
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))

# 训练集和测试集划分
train_data = true_data[:int(train_size * len(true_data))]
test_data = true_data[-int(test_size * len(true_data)):]
print("训练集尺寸:", len(train_data))
print("测试集尺寸:", len(test_data))

# 进行标准化处理
train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))

# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)


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


class GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=dbo_hidden, num_layers=2, output_dim=1, pre_len=1):
        super(GRU, self).__init__()
        self.pre_len = pre_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # 替换 LSTM 为 GRU
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        # self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dbo_dropout)

    def forward(self, x):
        h0_gru = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)
        # h0_gru = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0_gru)

        out = self.dropout(out)

        # 取最后 pre_len 时间步的输出
        out = out[:, -self.pre_len:, :]

        out = self.fc(out)
        out = self.relu(out)
        return out


gru_model = GRU(input_dim=1, output_dim=1, num_layers=2, hidden_dim=dbo_hidden, pre_len=pre_len)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=dbo_lr)
epochs = change_epoch

# 根据相关超参数进行模型训练
losss = []

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 开始训练GRU模型!")

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
torch.save(gru_model.state_dict(), tmp_path[dataset_choice] + '/save_model_gru.pth')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
      f"------ 模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")

# 根据训练结果模型进行预测
gru_model.load_state_dict(torch.load(tmp_path[dataset_choice] + '/save_model_gru.pth'))
gru_model.eval()  # 评估模式
results = []
reals = []
losss_mae = []
losss_rmse = []

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 利用GRU模型开始预测!")

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

real_and_pre_info = pd.DataFrame({'reals': reals, 'predict': results})
real_and_pre_info.to_csv(tmp_path[dataset_choice] + '/real_and_pre_info_gru.csv', index=False)

abs_diff = results - reals
print("****************************************")

# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 模型预测结果：", results)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 预测误差MAE:", losss_mae)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "------ 预测误差RMSE:", losss_rmse)
print("****************************************")

ave_mae = np.mean(losss_mae)
ave_rmse = np.mean(losss_rmse)

print("平均预测误差MAE：", ave_mae)
print("平均预测误差RMSE：", ave_rmse)
print("上述两平均预测误差之和：", ave_mae + ave_rmse)
abs_diff_round = [round(i, 3) for i in abs_diff]
# print("绝对差值：", abs_diff)
draw_28_types_pic_with_two_datalist(reals, results, tmp_path[dataset_choice])
'''
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
plt.savefig('./data_of_gru/test_results_gru.png')
plt.close('all')
'''
