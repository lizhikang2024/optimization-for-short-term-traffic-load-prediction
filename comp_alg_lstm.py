import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
import seaborn as sns

sns.set()
dataset_choice = 1  # 0 表示电网电荷流量数据， 1 表示英国学术数据中心网络流量数据
tmp_path = ['./data_of_lstm/etth', './data_of_lstm/ukdata']

# from torch.utils.data import Datasetpip


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
        plt.savefig(save_path + '/test_results_lstm_pic_' + str(i + 1) + '.png')
        plt.close('all')


np.random.seed(0)


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


"""
检查etth或ukdata文件夹是否存在于data_of_lstm文件夹中，不存在则新建，存在则清空子文件夹
"""
if os.path.exists(tmp_path[dataset_choice]):
    clear_folder(tmp_path[dataset_choice])
else:
    os.makedirs(tmp_path[dataset_choice])

"""
数据定义部分
"""
if dataset_choice == 0:
    true_data = pd.read_csv('ETTh1.csv', nrows=16000)  # 填你自己的数据地址,自动选取你最后一列数据为特征列
    # true_data = pd.read_csv('ETTh1.csv')  # 填你自己的数据地址,自动选取你最后一列数据为特征列
    target = 'OT'  # 添加你想要预测的特征列
else:
    true_data = pd.read_csv('ukdata.csv', nrows=16000)  # 填你自己的数据地址,自动选取你最后一列数据为特征列
    # true_data = pd.read_csv('ETTh1.csv')  # 填你自己的数据地址,自动选取你最后一列数据为特征列
    target = 'data'  # 添加你想要预测的特征列

# 这里加一些数据的预处理, 最后需要的格式是pd.series

true_data = np.array(true_data[target])

"""
LSTM超参数设置
"""
lstm_window = 24
lstm_hidden = 32
lstm_lr = 0.001
# epoch数
change_epoch = 20
print(torch.__version__)
# 定义窗口大小
test_data_size = lstm_window
# 训练集和测试集的尺寸划分
test_size = 0.2
train_size = 0.8
# 标准化处理
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))
train_data = true_data[:int(train_size * len(true_data))]
test_data = true_data[-int(test_size * len(true_data)):]
print("训练集尺寸:", len(train_data))
print("测试集尺寸:", len(test_data))
train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))
# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)


def create_inout_sequences(input_data, tw, pre_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + 4) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


pre_len = 1
train_window = lstm_window
# 定义训练器的的输入
train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)


class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=lstm_hidden, output_dim=1):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        h0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)
        c0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0_lstm, c0_lstm))
        out = out[:, -1]
        out = self.fc(out)

        return out


lstm_model = LSTM(input_dim=1, output_dim=pre_len, hidden_dim=train_window)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lstm_lr)
epochs = change_epoch

losss = []
lstm_model.train()  # 训练模式
start_time = time.time()  # 计算起始时间
for i in range(epochs):
    for seq, labels in train_inout_seq:
        lstm_model.train()
        optimizer.zero_grad()

        y_pred = lstm_model(seq)

        single_loss = loss_function(y_pred, labels)

        single_loss.backward()
        optimizer.step()
        # print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        losss.append(single_loss.detach().numpy())

if dataset_choice == 0:
    torch.save(lstm_model.state_dict(), './data_of_lstm/etth/save_model_lstm.pth')
else:
    torch.save(lstm_model.state_dict(), './data_of_lstm/ukdata/save_model_lstm.pth')

print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")
'''
plt.plot(losss)
# 设置图表标题和坐标轴标签
plt.title('Training Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
# 保存图表到本地
plt.savefig('./data_of_lstm/training_error.png')
plt.close('all')
'''
# 加载模型进行预测
if dataset_choice == 0:
    lstm_model.load_state_dict(torch.load('./data_of_lstm/etth/save_model_lstm.pth'))
else:
    lstm_model.load_state_dict(torch.load('./data_of_lstm/ukdata/save_model_lstm.pth'))
lstm_model.eval()  # 评估模式
results = []
reals = []
losss = []
test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)
for seq, labels in test_inout_seq:
    pred = lstm_model(seq)[0].item()
    results.append(pred)
    mae = calculate_mae(pred, labels.detach().numpy())  # MAE误差计算绝对值(预测值  - 真实值)
    reals.append(labels.detach().numpy())
    losss.append(mae)

reals = scaler_test.inverse_transform(np.array(reals).reshape(1, -1))[0]
results = scaler_test.inverse_transform(np.array(results).reshape(1, -1))[0]

real_and_pre_info = pd.DataFrame({'reals': reals, 'predict': results})
if dataset_choice == 0:
    real_and_pre_info.to_csv("./data_of_lstm/etth/real_and_pre_info_lstm.csv", index=False)
    pic_save_path = './data_of_lstm/etth'
else:
    real_and_pre_info.to_csv("./data_of_lstm/ukdata/real_and_pre_info_lstm.csv", index=False)
    pic_save_path = './data_of_lstm/ukdata'

print("模型预测结果：", results)
print("预测误差MAE:", losss)
draw_28_types_pic_with_two_datalist(reals, results, pic_save_path)
'''
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
plt.savefig('./data_of_lstm/test_results_lstm.png')
plt.close('all')
'''
