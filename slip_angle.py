import lowpassfilter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():

    output_data = pd.read_csv(
        "LOG00289.txt",
        header=None,
        delim_whitespace=False,
        # 古い引数の代わりに新しい引数を使う
        on_bad_lines="skip",
        # error_bad_lines=False  <- これは削除する
    )
    print(output_data)
    for col in output_data.columns:
        output_data[col] = pd.to_numeric(output_data[col], errors="coerce")
    # 時間軸（最初の列）に NaN がある行を削除する
    # 他のデータ列に NaN が残ってもプロットは可能ですが、時間軸は連続している必要があるため
    output_data = output_data.dropna(subset=[0]).reset_index(drop=True)

     # -----------

    output_data.iloc[:, 1] = lowpassfilter.lowpass_filter(
        output_data.iloc[:, 1], 1, 0.01
    )
    output_data.iloc[:, 2] = lowpassfilter.lowpass_filter(
        output_data.iloc[:, 2], 1, 0.01
    )
    output_data.iloc[:, 4] = lowpassfilter.lowpass_filter(
        output_data.iloc[:, 4], 1, 0.01
    )
    output_data.iloc[:, 5] = lowpassfilter.lowpass_filter(
        output_data.iloc[:, 5], 1, 0.01
    )
    slipangle(output_data)

def slipangle(output_data_):
    time_s = output_data_.iloc[:, 0] / 1000
    rad_vel_ave = lowpassfilter.lowpass_filter((output_data_.iloc[:, 3] + output_data_.iloc[:, 6]) / 2, 1, 0.01)
    rad_acc_ave = np.gradient(rad_vel_ave, time_s)
    filtered_rad_acc = lowpassfilter.lowpass_filter(rad_acc_ave, 1, 0.01)
    plt.figure(1)
    plt.plot(time_s, filtered_rad_acc, label='Filtered Radial Acceleration')
    plt.plot(time_s, rad_acc_ave, label='Raw Radial Acceleration', alpha=0.5)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Radial Acceleration')



    plt.show()
if __name__ == "__main__":

    main()
