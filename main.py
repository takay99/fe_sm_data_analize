import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import all_plot
from matplotlib.widgets import CheckButtons


def main():
    print("Hello from fe-sm-data-analize!")
    # df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    output_data = pd.read_csv('LOG00281.txt',header=None, delim_whitespace=False)

    # print(output_data.iloc[0,:])
    output_data = output_data.iloc[:,0:14]
    output_data = output_data.dropna(how='any')
    output_data.columns = ['tim', 'acc x r', 'acc y r', 'gyr z r', 'acc x f', 'acc y f', 'gyr z r', 'm_acc 1', 'm_acc 2',  'm vel 1', 'm vel 2', 'str' ,'currenrt1','currenrt2']
    output_data['tim'] = output_data['tim'] - output_data['tim'].iloc[0]
    print(output_data)

    all_plot.all_plot(output_data)
    print("end")
    # time = output_data.iloc[:, 0] / 1000

    # # グラフウィンドウの作成
    # plt.figure(figsize=(10, 6))

    # # 最初のプロット: poly_acc_1
  
    # plt.plot(time, output_data.iloc[:, 1], label='output_data column 1') # output_data(:,3)
    # # 後続のプロット
    
    # plt.plot(time, output_data.iloc[:, 2], label='output_data column 2') # output_data(:,3)
    # plt.plot(time, output_data.iloc[:, 3], label='output_data column 4') # output_data(:,4)
    
    # plt.plot(time, output_data.iloc[:, 4], label='output_data column 3') # output_data(:,3)
    # plt.plot(time, output_data.iloc[:, 5], label='output_data column 6') # output_data(:,6)
    # plt.plot(time, output_data.iloc[:, 6], label='output_data column 7') # output_data(:,7)
    # plt.plot(time, output_data.iloc[:, 7], label='output_data column 8') # output_data(:,8)
    # plt.plot(time, -output_data.iloc[:, 8], label='-output_data column 9') # -output_data(:,9)
    # plt.plot(time, -output_data.iloc[:, 9], label='-output_data column 10') # -output_data(:,10)
    # plt.plot(time, output_data.iloc[:, 10], label='output_data column 11') # output_data(:,11)
    # plt.plot(time, output_data.iloc[:, 11], label='output_data column 12') # output_data(:,12)
    # plt.plot(time, -output_data.iloc[:, 12] / 100, label='-output_data column 13/100') # -output_data(:,13)/100
    # plt.plot(time, output_data.iloc[:, 13] / 100, label='output_data column 14/100') # output_data(:,14)/100

    # # グリッドの表示
    # plt.grid(True)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Value')
    # plt.title('Data Plot')
    # plt.legend()
    # plt.show()

def main():
    print("Hello from fe-sm-data-analize!")
    # df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df = pd.read_csv('LOG00281.txt',header=None, delim_whitespace=False)

    print(df)
    print(df.iloc[0,:])
    df = df.iloc[:,0:14]
    df = df.dropna(how='any')
    # df.columns = ['新しい列名1', '新しい列名2', ...]
    print(df)
    all_plot.all_plot(df)

if __name__ == "__main__":
    main()
