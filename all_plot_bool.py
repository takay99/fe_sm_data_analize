import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons

def all_plot_bool(output_data):
    time = output_data.iloc[:, 0] / 1000


    # グラフウィンドウの作成
    plt.figure(figsize=(10, 6))

    # 最初のプロット: poly_acc_1
    plt.plot(time, output_data.iloc[:, 1], label='acc x r') # output_data(:,3)
    # 後続のプロット
    
    plt.plot(time, output_data.iloc[:, 2], label='acc y r') # output_data(:,3)
    plt.plot(time, output_data.iloc[:, 3], label='gyr z r') # output_data(:,4)
    plt.plot(time, output_data.iloc[:, 4], label='acc x f') # output_data(:,3)
    plt.plot(time, output_data.iloc[:, 5], label='acc y f') # output_data(:,6)
    plt.plot(time, output_data.iloc[:, 6], label='gyr z f') # output_data(:,7)
    plt.plot(time, output_data.iloc[:, 7], label='m acc 1') # output_data(:,8)
    plt.plot(time, -output_data.iloc[:, 8], label='m acc 2') # -output_data(:,9)
    plt.plot(time, -output_data.iloc[:, 9], label='m vel 1') # -output_data(:,10)
    plt.plot(time, output_data.iloc[:, 10], label='m vel 2') # output_data(:,11)
    plt.plot(time, output_data.iloc[:, 11], label='str') # output_data(:,12)
    plt.plot(time, -output_data.iloc[:, 12] / 100, label='current1') # -output_data(:,13)/100
    plt.plot(time, output_data.iloc[:, 13] / 100, label='current2') # output_data(:,14)/100

    # グリッドの表示
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Data Plot')
    plt.legend()
    plt.show()

