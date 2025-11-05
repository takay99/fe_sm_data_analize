import lowpassfilter
import pandas as pd


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

    slipangle(output_data)


def slipangle(output_data_):
    filtered_rad_vel = lowpassfilter.lowpass_filter(output_data_.iloc[:, 3], 1, 0.01)
