# 본 소스는 3번 문제에 대한 plot을 그리기 위해서 존재하는 모듈입니다.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import random
import math


def plot_3d(np_data, GENE):
    # pos sample
    x_1 = np_data[0:50, 0]
    y_1 = np_data[0:50, 1]
    z_1 = np_data[0:50, 2]

    # neg sample
    x_0 = np_data[51:, 0]
    y_0 = np_data[51:, 1]
    z_0 = np_data[51:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    w1, w2, w3, b = GENE
    ax.plot(x_1, y_1, z_1, linestyle="none", marker="o", mfc="none", markeredgecolor="b")
    ax.plot(x_0, y_0, z_0, linestyle="none", marker="o", mfc="none", markeredgecolor="r")

    X = np.arange(0, 2, 0.1) * 100
    Y = np.arange(0, 2, 0.1) * 100
    X, Y = np.meshgrid(X, Y)

    Z = (-float(w1)/w3 * X) + (-float(w2)/w3 * Y) - float(b)/w3 # 평면의 방정식
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.4, cmap=cm.Blues) # 평면 출력
    plt.show()