import numpy as np
import function as fun
import DBO as fun1
import sys
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path


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


def main(argv):
    SearchAgents_no = 30
    Function_name = 'F24'
    Max_iteration = 150

    """
    清空GRU模型保存文件夹和GRU预测结果图文件夹
    """
    if Function_name == 'F24':
        folder_path = "./data_of_dbo-gru/result_model"  # GRU模型保存文件夹路径
        clear_folder(folder_path)  # 清空文件夹内的所有文件
        folder_path = "./data_of_dbo-gru/result_pic"  # GRU预测结果图文件夹路径
        clear_folder(folder_path)  # 清空文件夹内的所有文件
        folder_path = "./data_of_dbo-gru/result_csv"  # GRU预测结果csv文件路径
        clear_folder(folder_path)  # 清空文件夹内的所有文件

    print('++++++++++++++++ Start DBO-main ++++++++++++++++')
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print('\n\n\n\n', file=plog_write)
    print('++++++++++++++++ Start DBO-main ++++++++++++++++', file=plog_write)
    plog_write.close()

    [fobj, lb, ub, dim] = fun1.Parameters(Function_name)
    [fMin, bestX, DBO_curve] = fun1.DBO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)

    print(['最优值为：', fMin])
    print(['最优变量为：', bestX])
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '------', '最优值为：', fMin, file=plog_write)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '------', '最优变量为：', bestX,
          file=plog_write)
    plog_write.close()

    print('++++++++++++++++ End DBO-main  ++++++++++++++++')
    # 记录信息到日志文件中
    plog_write = open('DBO_plog.txt', 'a')
    print('++++++++++++++++ End DBO-main  ++++++++++++++++', file=plog_write)
    plog_write.close()


'''
    thr1 = np.arange(len(DBO_curve[0, :]))

    plt.plot(thr1, DBO_curve[0, :])

    plt.xlabel('num')
    plt.ylabel('object value')
    plt.title('line')

    plt.show()
'''

if __name__ == '__main__':
    main(sys.argv)
