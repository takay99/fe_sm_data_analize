import numpy as np


def lowpass_filter(data, cutoff_freq, T):
    data = np.asarray(data)
    omega_c_T = cutoff_freq * 2 * np.pi * T
    alpha = omega_c_T / (1 + omega_c_T)
    D = 2 + cutoff_freq * T
    # a1 = (cutoff_freq * T - 2) / D
    # b1 = cutoff_freq * T / D
    # b0 = b1
    y = np.zeros(len(data))
    for i in range(1, len(data)):
        # y[i] = a1 * y[i - 1] + b0 * data[i] + b1 * data[i - 1]
        y[i] = y[i - 1] + alpha * (data[i] - y[i - 1])
    return y
