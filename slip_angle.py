import lowpassfilter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import all_plot_bool
import math


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
    slip_angle_plot = slipangle(output_data)
    fig_all, ax_all = all_plot_bool.all_plot_bool(output_data)
    plt.show()


def slipangle(output_data_):
    time_s = output_data_.iloc[:, 0] / 1000
    rad_vel_ave = lowpassfilter.lowpass_filter(
        (output_data_.iloc[:, 3] + output_data_.iloc[:, 6]) / 2, 1, 0.01
    )
    rad_acc_ave = np.gradient(rad_vel_ave, time_s)
    filtered_rad_acc = lowpassfilter.lowpass_filter(rad_acc_ave, 1, 0.01)

    acc_f_outradacc = output_data_.iloc[:, 4] - rad_acc_ave * (0.23 / 2)
    acc_r_outradacc = output_data_.iloc[:, 1] + rad_acc_ave * (0.23 / 2)

    acc_f = np.hypot(acc_f_outradacc, output_data_.iloc[:, 5])
    acc_r = np.hypot(acc_r_outradacc, output_data_.iloc[:, 2])

    radius_f = acc_f / np.hypot(rad_vel_ave**2, rad_acc_ave)
    radius_r = acc_r / np.hypot(rad_vel_ave**2, rad_acc_ave)

    # array_radius_f = lowpassfilter.lowpass_filter(pd.Series(radius_f), 0.05, 0.01)
    # array_radius_r = lowpassfilter.lowpass_filter(pd.Series(radius_r), 0.05, 0.01)
    array_radius_f = pd.Series(radius_f)
    array_radius_r = pd.Series(radius_r)
    array_rad_vel = pd.Series(rad_vel_ave)

    mask = np.abs(array_rad_vel) > 0.3
    array_radius_f = array_radius_f[mask]
    array_radius_r = array_radius_r[mask]

    print(mask)

    # array_radius_f = conditionally_flip_sign(array_rad_vel, array_radius_f)
    # array_radius_r = conditionally_flip_sign(array_rad_vel, array_radius_r)

    result = calc_af_ar_series(array_radius_f, array_radius_r)
    front_slip = np.arctan(result.iloc[:, 1] / result.iloc[:, 0])
    rear_slip = np.arctan(result.iloc[:, 3] / result.iloc[:, 2])

    centripetal_acc_r =np.sign(array_rad_vel[mask])*array_radius_r* array_rad_vel[mask] ** 2 #- rad_acc_ave[mask] * (0.23 / 2)
    lateral_friction_coeff_r = centripetal_acc_r / ((9.81/2) + ((2*0.045/0.23)) )
    

    fig_1, ax_1 = plt.subplots()
    ax_1.plot(time_s[mask], array_radius_f, label="Filtered Rsadial Acceleration")
    ax_1.plot(time_s[mask], array_radius_r, label="Raw Radial Acceleration", alpha=0.5)
    ax_1.grid(True)
    ax_1.set_xlabel("Time (s)")
    ax_1.set_ylabel("Radial Acceleration")

    fig_2, ax_2 = plt.subplots()
    # plt.figure(1)
    ax_2.plot(time_s[mask], front_slip, "o", label="Filtered Radial Acceleration")
    ax_2.plot(time_s[mask], rear_slip, "o", label="Raw Radial Acceleration", alpha=0.5)
    # ax_2.plot(
    #     time_s[mask], result.iloc[:, 2], label="Raw Radial Acceleration", alpha=0.5
    # )
    # ax_2.plot(
    #     time_s[mask], result.iloc[:, 3], label="Raw Radial Acceleration", alpha=0.5
    # )
    ax_2.grid(True)
    ax_2.set_xlabel("Time (s)")

    fig_3, ax_3 = plt.subplots()
    ax_3.plot(
        time_s[mask], result.iloc[:, 2], label="Raw Radial Acceleration", alpha=0.5
    )
    ax_3.plot(
        time_s[mask], result.iloc[:, 3], label="Raw Radial Acceleration", alpha=0.5
    )
    ax_3.plot(
        time_s[mask], result.iloc[:, 0], label="Raw Radial Acceleration", alpha=0.5
    )
    ax_3.plot(
        time_s[mask], result.iloc[:, 1], label="Raw Radial Acceleration", alpha=0.5
    )
    ax_3.grid(True)
    ax_3.set_xlabel("Time (s)")

    fig_4, ax_4 = plt.subplots()
    ax_4.plot(rear_slip,lateral_friction_coeff_r,'o', label="Slip Angle vs Lateral Friction Coeff")
    ax_4.grid(True)
    ax_4.set_xlabel("Slip Angle (rad)")
    ax_4.set_ylabel("Lateral Friction Coeff")   

    return (
        (fig_1, ax_1),
        (fig_2, ax_2),
        (fig_3, ax_3),
        (fig_4, ax_4),
    )

    # plt.show()


def conditionally_flip_sign(eval_array, target_array):
    """
    評価配列の値が負の場合、操作対象配列の対応する要素の符号を反転させる関数。

    Args:
        eval_array (np.ndarray): 正負を判断するための配列。
        target_array (np.ndarray): 符号を変更する操作対象の配列。

    Returns:
        np.ndarray: 符号が変更された後の操作対象配列。
    """
    # 1. マスクを作成: 評価配列が負である要素を True とする
    # 例: eval_array = [ 1, -2,  3, -4] の場合、mask = [False, True, False, True]
    mask_negative = eval_array < 0

    # 2. 符号を変更する対象の要素（Trueの要素）のみ符号を反転させる
    # 元の target_array[mask_negative] に -1 を掛ける
    # ※ この操作は元の target_array を直接変更します
    target_array[mask_negative] *= -1

    # 変更後の配列を返す
    return target_array


def calc_af_ar_series(array_radius_f, array_radius_r):
    """
    array_radius_f, array_radius_r: 同じ長さの NumPy array または pandas.Series
    戻り値: DataFrame(columns=["afx", "afy", "arx", "ary"])
    """

    r_f = np.asarray(array_radius_f)
    r_r = np.asarray(array_radius_r)
    d = 0.23  # 前後の距離差 [m]

    # a_ry を計算
    ary = (r_f**2 - r_r**2 - d**2) / (2 * d)

    # sqrtの中身が負になる場合はNaN（実数解なし）
    inside = r_r**2 - ary**2
    inside[inside < 0] = np.nan

    arx = np.sqrt(inside)
    afx = arx
    afy = ary + d

    # DataFrameでまとめて返す
    return pd.DataFrame({"afx": afx, "afy": afy, "arx": arx, "ary": ary})


if __name__ == "__main__":

    main()
