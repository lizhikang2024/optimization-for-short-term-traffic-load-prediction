import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from pathlib import Path
import seaborn as sns

sns.set()
dataset_choice = 1  # 0 表示电网电荷流量数据， 1 表示英国学术数据中心网络流量数据
tmp_path = ['./data_of_tcn/etth', './data_of_tcn/ukdata']


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
        plt.savefig(save_path + '/test_results_tcn_pic_' + str(i + 1) + '.png')
        plt.close('all')


# 随机数种子
np.random.seed(0)


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def plot_loss_data(data):
    # 使用Matplotlib绘制线图
    plt.figure()

    plt.plot(data, marker='o')

    # 添加标题
    plt.title("loss results Plot")

    # 显示图例
    plt.legend(["Loss"])

    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path, nrows=16000)  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口

    # 将特征列移到末尾
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values

    # 定义标准化优化器
    # 定义标准化优化器
    scaler = StandardScaler()
    scaler.fit(true_data)

    train_data = true_data[:int(0.8 * len(true_data))]
    test_data = true_data[-int(0.2 * len(true_data)):]
    print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data))

    '''
    train_data = true_data[int(0.3 * len(true_data)):]
    valid_data = true_data[int(0.15 * len(true_data)):int(0.30 * len(true_data))]
    test_data = true_data[:int(0.15 * len(true_data))]
    print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))
    '''

    # 进行标准化处理
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    # valid_data_normalized = scaler.transform(valid_data)

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    # valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    # valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    # valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
    # print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # return train_loader, test_loader, valid_loader, scaler
    return train_loader, test_loader, scaler


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, outputs, pre_len, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.pre_len = pre_len

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x[:, -self.pre_len:, :]


def train(model, args, scaler, device):
    start_time = time.time()  # 计算起始时间
    model = model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = args.epochs
    model.train()  # 训练模式
    results_loss = []
    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)

            single_loss.backward()

            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss))

        torch.save(model.state_dict(), tmp_path[dataset_choice]+'/save_model_tcn.pth')
        time.sleep(0.1)

    # valid_loss = valid(model, args, scaler, valid_loader)
    # 尚未引入学习率计划后期补上
    # 保存模型

    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    # plot_loss_data(results_loss)


'''
def valid(model, args, scaler, valid_loader):
    lstm_model = model
    # 加载模型进行预测
    lstm_model.load_state_dict(torch.load('save_model.pth'))
    lstm_model.eval()  # 评估模式
    losss = []

    for seq, labels in valid_loader:
        pred = lstm_model(seq)
        mae = calculate_mae(pred.detach().numpy().cpu(), np.array(labels.detach().cpu()))  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)

    print("验证集误差MAE:", losss)
    return sum(losss) / len(losss)
'''


def test(model, args, test_loader, scaler):
    # 加载模型进行预测
    losss = []
    model = model
    model.load_state_dict(torch.load(tmp_path[dataset_choice]+'/save_model_tcn.pth'))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        mae = calculate_mae(pred.detach().cpu().numpy(),
                            np.array(label.detach().cpu()))  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)
        pred = pred[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])

    real_and_pre_info = pd.DataFrame({'reals': labels, 'predict': results})
    real_and_pre_info.to_csv(tmp_path[dataset_choice]+'/real_and_pre_info_tcn.csv', index=False)

    print("测试集误差MAE:", losss)
    draw_28_types_pic_with_two_datalist(labels, results, tmp_path[dataset_choice])
    '''
    # 绘制历史数据
    plt.plot(labels, label='TrueValue')

    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(results, label='Prediction')

    # 添加标题和图例
    plt.title("test state")
    plt.legend()
    plt.savefig('./data_of_tcn/test_results_tcn.png')
    plt.close('all')
    # plt.show()
    '''

# 检验模型拟合情况
def inspect_model_fit(model, args, train_loader, scaler):
    model = model
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()  # 评估模式
    results = []
    labels = []

    for seq, label in train_loader:
        pred = model(seq)[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])

    # 绘制历史数据
    plt.plot(labels, label='History')

    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(results, label='Prediction')

    # 添加标题和图例
    plt.title("inspect model fit state")
    plt.legend()
    plt.show()


def predict(model, args, device, scaler):
    # 预测未知数据的功能
    df = pd.read_csv(args.data_path)
    df = df.iloc[:, 1:][-args.window_size:].values  # 转换为nadarry
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()  # 评估模式

    pred = model(tensor_pred)[0]

    pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    # 假设 df 和 pred 是你的历史和预测数据

    # 计算历史数据的长度
    history_length = len(df[:, -1])

    # 为历史数据生成x轴坐标
    history_x = range(history_length)

    # 为预测数据生成x轴坐标
    # 开始于历史数据的最后一个点的x坐标
    prediction_x = range(history_length - 1, history_length + len(pred[:, -1]) - 1)

    # 绘制历史数据
    plt.plot(history_x, df[:, -1], label='History')

    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(prediction_x, pred[:, -1], marker='o', label='Prediction')
    plt.axvline(history_length - 1, color='red')  # 在图像的x位置处画一条红色竖线
    # 添加标题和图例
    plt.title("History and Prediction")
    plt.legend()


if __name__ == '__main__':

    """
    检查etth或ukdata文件夹是否存在于data_of_tcn文件夹中，不存在则新建，存在则清空子文件夹
    """
    if os.path.exists(tmp_path[dataset_choice]):
        clear_folder(tmp_path[dataset_choice])
    else:
        os.makedirs(tmp_path[dataset_choice])

    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=24, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=1, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")

    if dataset_choice == 0:
        parser.add_argument('-data_path', type=str, default='ETTh1.csv', help="你的数据数据地址")
        parser.add_argument('-target', type=str, default='OT', help='你需要预测的特征列，这个值会最后保存在csv文件里')
        parser.add_argument('-input_size', type=int, default=7, help='你的特征个数不算时间那一列')
    else:
        parser.add_argument('-data_path', type=str, default='ukdata.csv', help="你的数据数据地址")
        parser.add_argument('-target', type=str, default='data', help='你需要预测的特征列，这个值会最后保存在csv文件里')
        parser.add_argument('-input_size', type=int, default=1, help='你的特征个数不算时间那一列')

    parser.add_argument('-feature', type=str, default='S', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    parser.add_argument('-model_dim', type=list, default=[64, 128, 256], help='这个地方是这个TCN卷积的关键部分,它代表了TCN的层数我这里输'
                                                                              '入list中包含三个元素那么我的TCN就是三层，这个根据你的数据复杂度来设置'
                                                                              '层数越多对应数据越复杂但是不要超过5层')

    # learning
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=1, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=24, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')

    # model
    parser.add_argument('-hidden_size', type=int, default=32, help="隐藏层单元数")
    parser.add_argument('-kernel_sizes', type=int, default=3)
    parser.add_argument('-laryer_num', type=int, default=1)
    # device
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)
    args = parser.parse_args()

    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("使用设备:", device)
    train_loader, test_loader, scaler = create_dataloader(args, device)

    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = TemporalConvNet(args.input_size, args.output_size, args.pre_len, args.model_dim, args.kernel_sizes).to(
            device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # 训练模型
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device)
    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)
'''
    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler)
    plt.show()
'''
