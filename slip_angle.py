import lowpassfilter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import all_plot_bool
import math
import csv


def main():

    output_data = pd.read_csv(
        "LOG00295.txt",
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


# def slipangle(output_data_):
#     time_s = output_data_.iloc[:, 0] / 1000
#     rad_vel_ave = lowpassfilter.lowpass_filter(
#         (output_data_.iloc[:, 3] + output_data_.iloc[:, 6]) / 2, 1, 0.01
#     )
#     rad_acc_ave = np.gradient(rad_vel_ave, time_s)
#     filtered_rad_acc = lowpassfilter.lowpass_filter(rad_acc_ave, 1, 0.01)

#     acc_f_outradacc = output_data_.iloc[:, 4] - rad_acc_ave * (0.23 / 2)
#     acc_r_outradacc = output_data_.iloc[:, 1] + rad_acc_ave * (0.23 / 2)

#     acc_f = np.hypot(acc_f_outradacc, output_data_.iloc[:, 5])
#     acc_r = np.hypot(acc_r_outradacc, output_data_.iloc[:, 2])

#     radius_f = acc_f / np.hypot(rad_vel_ave**2, rad_acc_ave)
#     radius_r = acc_r / np.hypot(rad_vel_ave**2, rad_acc_ave)

#     array_radius_f = lowpassfilter.lowpass_filter(pd.Series(radius_f), 1, 0.01)
#     array_radius_r = lowpassfilter.lowpass_filter(pd.Series(radius_r), 1, 0.01)
#     # array_radius_f = pd.Series(radius_f)
#     # array_radius_r = pd.Series(radius_r)
#     array_rad_vel = pd.Series(rad_vel_ave)

#     mask = np.abs(array_rad_vel) > 0.1
#     array_radius_f = array_radius_f[mask]
#     array_radius_r = array_radius_r[mask]

#     print(mask)

#     # array_radius_f = conditionally_flip_sign(array_rad_vel, array_radius_f)
#     # array_radius_r = conditionally_flip_sign(array_rad_vel, array_radius_r)

#     result = calc_af_ar_series(array_radius_f, array_radius_r)

#     # x_r = lowpassfilter.lowpass_filter(result.iloc[:, 2], 100, 0.01)
#     x_r = result.iloc[:, 2]
#     # y_r = lowpassfilter.lowpass_filter(result.iloc[:, 3], 1, 0.01)
#     y_r = lowpassfilter.lowpass_filter(result.iloc[:, 3], 1, 0.01)
#     x_f = lowpassfilter.lowpass_filter(result.iloc[:, 0], 1, 0.01)
#     y_f = lowpassfilter.lowpass_filter(result.iloc[:, 1], 1, 0.01)

#     # 1) 半径角（rear, front）を atan2(lateral, longitudinal) で求める
#     angle_radius_rear = np.arctan2(y_r, x_r)  # atan2(lateral, longitudinal)
#     angle_radius_front = np.arctan2(y_f, x_f)

#     # 2) 接線（速度）方向 = 半径角 + sign * (pi/2)
#     # sign は角速度（ヨー）に合わせる。ヨーが正のとき回転方向に依存して符号を決定。
#     sign = np.sign(array_rad_vel[mask])  # 既に mask を適用済みのものを想定

#     tangent_angle_rear = angle_radius_rear * sign  # * (np.pi / 2)
#     tangent_angle_front = angle_radius_front * sign  # * (np.pi / 2)

#     # 3) スリップ角 = 接線方向 - (ホイール向き)
#     # - 後輪: wheel heading = 車体前向き（assume 0） → rear_slip = tangent_angle_rear
#     # - 前輪: wheel heading = steering angle (delta) があれば差し引く
#     #   ここでは steering_angle_series を持っている前提で示す（なければ 0 を使う）
#     steering_angle = np.zeros_like(tangent_angle_front)  # もしデータ列があれば置き換え
#     rear_slip = tangent_angle_rear  # - vehicle_heading if needed
#     front_slip = tangent_angle_front  # - steering_angle

#     centripetal_acc_r = (
#         np.sign(array_rad_vel[mask]) * array_radius_r * array_rad_vel[mask] ** 2
#     )  # - rad_acc_ave[mask] * (0.23 / 2)
#     lateral_friction_coeff_r = centripetal_acc_r / ((9.81 / 2) + ((2 * 0.045 / 0.23)))

#     # ここにNANや定義したマスクを時系列データをNANを無くして時系列が崩れニア用に整理するコードを改定

#     fig_1, ax_1 = plt.subplots()
#     ax_1.plot(time_s[mask], array_radius_f, label="Filtered Rsadial Acceleration")
#     ax_1.plot(time_s[mask], array_radius_r, label="Raw Radial Acceleration", alpha=0.5)

#     ax_1.grid(True)
#     ax_1.set_xlabel("Time (s)")
#     ax_1.set_ylabel("Radial Acceleration")

#     fig_2, ax_2 = plt.subplots()
#     # plt.figure(1)
#     # ax_2.plot(
#     #     time_s[mask],
#     #     front_slip,
#     #     "o",
#     #     label="Filtered Radial Acceleration",
#     # )
#     ax_2.plot(
#         time_s[mask],
#         rear_slip,
#         "o",
#         label="Raw Radial Acceleration",
#         alpha=0.5,
#     )
#     # ax_2.plot(
#     #     time_s[mask], result.iloc[:, 2], label="Raw Radial Acceleration", alpha=0.5
#     # )
#     # ax_2.plot(
#     #     time_s[mask], result.iloc[:, 3], label="Raw Radial Acceleration", alpha=0.5
#     # )
#     ax_2.grid(True)
#     ax_2.set_xlabel("Time (s)")

#     fig_3, ax_3 = plt.subplots()
#     ax_3.plot(time_s[mask], x_r, label="Raw Radial Acceleration", alpha=0.5)
#     ax_3.plot(
#         time_s[mask], result.iloc[:, 2], label="Raw Radial Acceleration", alpha=0.5
#     )
#     # ax_3.plot(time_s[mask], y_r, label="Raw Radial Acceleration", alpha=0.5)
#     # ax_3.plot(
#     #     time_s[mask], result.iloc[:, 0], label="Raw Radial Acceleration", alpha=0.5
#     # )
#     # ax_3.plot(
#     #     time_s[mask], result.iloc[:, 1], label="Raw Radial Acceleration", alpha=0.5
#     # )
#     ax_3.grid(True)
#     ax_3.set_xlabel("Time (s)")

#     fig_4, ax_4 = plt.subplots()
#     ax_4.plot(
#         rear_slip,
#         lateral_friction_coeff_r,
#         "o",
#         label="Slip Angle vs Lateral Friction Coeff",
#     )
#     ax_4.grid(True)
#     ax_4.set_xlabel("Slip Angle (rad)")
#     ax_4.set_ylabel("Lateral Friction Coeff")

#     return (
#         (fig_1, ax_1),
#         (fig_2, ax_2),
#         (fig_3, ax_3),
#         (fig_4, ax_4),
#     )

# plt.show()


def slipangle(output_data_):
    """
    入力データからスリップ角や横摩擦係数を算出し、
    4つのグラフ (fig, ax) を返す。
    """

    # 1. データの準備と初期計算
    time_s = output_data_.iloc[:, 0] / 1000
    rad_vel_ave = lowpassfilter.lowpass_filter(
        (output_data_.iloc[:, 3] + output_data_.iloc[:, 6]) / 2, 1, 0.01
    )
    rad_acc_ave = np.gradient(rad_vel_ave, time_s)
    # filtered_rad_acc = lowpassfilter.lowpass_filter(rad_acc_ave, 1, 0.01) # 未使用

    acc_f_outradacc = output_data_.iloc[:, 4] - rad_acc_ave * (0.23 / 2)
    acc_r_outradacc = output_data_.iloc[:, 1] + rad_acc_ave * (0.23 / 2)

    acc_f = np.hypot(acc_f_outradacc, output_data_.iloc[:, 5])
    acc_r = np.hypot(acc_r_outradacc, output_data_.iloc[:, 2])

    radius_f = acc_f / np.hypot(rad_vel_ave**2, rad_acc_ave)
    radius_r = acc_r / np.hypot(rad_vel_ave**2, rad_acc_ave)

    array_radius_f = lowpassfilter.lowpass_filter(pd.Series(radius_f), 1, 0.01)
    array_radius_r = lowpassfilter.lowpass_filter(pd.Series(radius_r), 1, 0.01)
    array_rad_vel = pd.Series(rad_vel_ave)

    # rad_acc_ave も Pandas Series にしておく
    s_rad_acc_ave = pd.Series(rad_acc_ave)

    # 2. マスクの作成と適用 (インデックスのリセット)
    # 角速度が小さい（ほぼ停止している）データを除外
    mask_series = np.abs(array_rad_vel) > 0.01
    # (A) 元々 pd.Series だった変数は、.reset_index() が必要
    time_s_masked = time_s[mask_series].reset_index(drop=True)
    array_rad_vel_masked = array_rad_vel[mask_series].reset_index(drop=True)
    # s_rad_acc_ave_masked = s_rad_acc_ave[mask_series].reset_index(drop=True) # 必要なら

    # (B) lowpassfilter が返した np.ndarray の場合、.reset_index() は不要

    # np.ndarray をフィルタリングするために、mask_series から .values を使って
    # ブール値の np.ndarray を取り出す
    mask_np = mask_series.values

    # np.ndarray には mask_np を適用するだけ
    array_radius_f_masked = array_radius_f[mask_np]
    array_radius_r_masked = array_radius_r[mask_np]

    y_term = lowpassfilter.lowpass_filter(
        (
            (rad_acc_ave[mask_np]) * (output_data_.iloc[:, 2][mask_np])
            - np.sign(rad_vel_ave[mask_np])
            * rad_vel_ave[mask_np] ** 2
            * (acc_r_outradacc[mask_np])
        ),
        1,
        0.01,
    )
    x_term = lowpassfilter.lowpass_filter(
        (
            (rad_acc_ave[mask_np]) * (acc_r_outradacc[mask_np])
            + np.sign(rad_vel_ave[mask_np])
            * rad_vel_ave[mask_np] ** 2
            * (output_data_.iloc[:, 2][mask_np])
        ),
        1,
        0.01,
    )
    rear_slip_2 = np.arctan2(y_term, x_term) + np.pi / 2  # <- 修正後のコード

    # rear_slip_2 = np.atan2(
    #     acc_r_outradacc[mask_np], (output_data_.iloc[:, 2][mask_np])
    # ) - np.atan2(rad_vel_ave[mask_np] ** 2, (rad_acc_ave[mask_np]))

    # --- 修正箇所 2: オフセットの計算と適用 ---
    # (fig_2 と fig_4 の両方を中央揃えするため、ここでオフセットを計算)
    # NaN が含まれることを考慮して nanmedian を使用
    offset = np.nanmedian(rear_slip_2)
    print(f"Calculated offset (median): {offset}")  # オフセット量を確認

    # オフセットを適用したスリップ角
    rear_slip_2_centered = rear_slip_2 - offset

    # --------------------S

    # 3. マスク適用後のデータで計算
    # (これ以降のコードは、修正した _masked 変数を使うため変更不要)SSS
    result = calc_af_ar_series(array_radius_f_masked, array_radius_r_masked)

    # result から各列を取得 (インデックスは 0..M-1)
    x_r = result.iloc[:, 2]
    y_r = result.iloc[:, 3]
    x_f = result.iloc[:, 0]
    y_f = result.iloc[:, 1]

    # 4. スリップ角の計算
    # 1) 半径角
    angle_radius_rear = np.arctan2(y_r, x_r)  # atan2(lateral, longitudinal)
    angle_radius_front = np.arctan2(y_f, x_f)

    # 2) 接線（速度）方向
    # (array_rad_vel_masked はインデックス 0..M-1 なので、そのまま計算できる)
    sign = np.sign(array_rad_vel_masked)

    tangent_angle_rear = angle_radius_rear * sign
    tangent_angle_front = angle_radius_front * sign

    # 3) スリップ角
    rear_slip = tangent_angle_rear
    # front_slip = tangent_angle_front # （前輪は今回未使用）

    # 5. 横摩擦係数の計算
    # rad_acc_ave は元データ長の配列なので、mask_np によってマスクして長さを揃える
    rad_acc_masked = rad_acc_ave[mask_np]

    centripetal_term = (
        np.sign(array_rad_vel_masked)
        * array_radius_r_masked
        * (array_rad_vel_masked ** 2)
        + rad_acc_masked * (0.23 / 2)
    )

    centripetal_acc_r = lowpassfilter.lowpass_filter(centripetal_term, 1, 0.01)
    lateral_friction_coeff_r = centripetal_acc_r / ((9.81 / 2) + ((2 * 0.045 / 0.23)))

    # 6. プロット用データフレームの作成
    # 計算過程で NaN が発生した場合 (例: calc_af_ar_series の sqrt が負) に備え、
    # .dropna() でそれらの行を除外する
    final_plot_data = (
        pd.DataFrame(
            {
                "time": time_s_masked,
                "r_f": array_radius_f_masked,
                "r_r": array_radius_r_masked,
                "x_r": x_r,
                "y_r": y_r,
                "rear_slip": rear_slip,
                # "rear_slip_2": rear_slip_2,
                "lat_fric_r": lateral_friction_coeff_r,
                "result_2": result.iloc[:, 2],  # フィルタ前の arx
            }
        )
        .dropna()  # NaN が含まれる行を削除
        .reset_index(drop=True)  # インデックスを再度振り直す
    )

    # 7. クリーンなデータをNumpy配列として取得
    time_c = final_plot_data["time"].to_numpy()
    r_f_c = final_plot_data["r_f"].to_numpy()
    r_r_c = final_plot_data["r_r"].to_numpy()
    rear_slip_c = final_plot_data["rear_slip"].to_numpy()
    # rear_slip_2_c = final_plot_data["rear_slip_2"].to_numpy()
    lat_fric_r_c = final_plot_data["lat_fric_r"].to_numpy()
    x_r_c = final_plot_data["x_r"].to_numpy()
    y_r_c = final_plot_data["y_r"].to_numpy()
    result_2_c = final_plot_data["result_2"].to_numpy()

    # 8. 4つのグラフを作成
    # Fig 1: 時間 vs 半径方向加速度
    fig_1, ax_1 = plt.subplots()
    ax_1.plot(time_c, r_f_c, label="Filtered Rsadial Acceleration (Front)")
    ax_1.plot(time_c, r_r_c, label="Filtered Radial Acceleration (Rear)", alpha=0.5)
    ax_1.grid(True)
    ax_1.set_xlabel("Time (s)")
    ax_1.set_ylabel("Radial Acceleration")
    ax_1.legend()  # 凡例を表示

    # Fig 2: 時間 vs 後輪スリップ角
    fig_2, ax_2 = plt.subplots()
    ax_2.plot(
        time_s[mask_np],
        rear_slip_2,
        "o",
        label="Rear Slip Angle",
        alpha=0.5,
    )
    ax_2.plot(
        time_s[mask_np],
        lateral_friction_coeff_r,
        "o",
        label="Rear Slip Angle",
        alpha=0.5,
    )
    ax_2.grid(True)
    ax_2.set_xlabel("Time (s)")
    ax_2.set_ylabel("Rear Slip Angle (rad)")
    ax_2.legend()

    # Fig 3: 時間 vs x_r (フィルタ処理前後比較)
    fig_3, ax_3 = plt.subplots()
    ax_3.plot(time_c, x_r_c, label="x_r (arx after LPF on ary)", alpha=0.5)
    ax_3.plot(time_c, y_r_c, label="Raw arx (before LPF)", alpha=0.5)
    ax_3.grid(True)
    ax_3.set_xlabel("Time (s)")
    ax_3.set_ylabel("x_r (longitudinal component)")
    ax_3.legend()

    # Fig 4: スリップ角 vs 横摩擦係数
    fig_4, ax_4 = plt.subplots()
    ax_4.plot(
        rear_slip_2,
        lateral_friction_coeff_r,
        "o",
        label="Slip Angle vs Lateral Friction Coeff",
    )
    ax_4.grid(True)
    ax_4.set_xlabel("Slip Angle (rad)")
    ax_4.set_ylabel("Lateral Friction Coeff")
    ax_4.legend()

    header = ["slipangle", "lateral_friction_coeff_r"]

    # --- 修正点 2 ---
    # zip() を使い、[ (角度1, 係数1), (角度2, 係数2), ... ] という
    # 「行のタプルのリスト」に変換します。
    rows = zip(rear_slip_2, lateral_friction_coeff_r)

    with open(
        "./slipangle_vs_lateralfrictioncoeff.csv",
        mode="w",
        newline="",
        encoding="UTF-8",
    ) as file:
        # (フィッティングのためには、タブ区切り(delimiter="\t")よりも
        #  カンマ区切り(デフォルト)の方が一般的です)
        writer = csv.writer(file)

        # --- 修正点 3 ---
        # ヘッダは writerow() (単数形) で1行書き込む
        writer.writerow(header)

        # --- 修正点 4 ---
        # zip()で作成した行データを writerows() (複数形) で全て書き込む
        writer.writerows(rows)
        # 9. グラフのタプルを返す
    return (
        (fig_1, ax_1),
        (fig_2, ax_2),
        (fig_3, ax_3),
        (fig_4, ax_4),
    )


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
