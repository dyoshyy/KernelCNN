import matplotlib.font_manager

del matplotlib.font_manager.weight_dict["roman"]
# matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"  # font familyの設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.size"] = 30  # 全体のフォントサイズが変更されます。
plt.rcParams["legend.fontsize"] = 8  # 凡例のフォントサイズ
plt.rcParams["xtick.labelsize"] = 25  # 軸だけ変更されます。
plt.rcParams["ytick.labelsize"] = 25  # 軸だけ変更されます
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid
plt.rcParams["legend.fancybox"] = False  # 丸角
plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1.2  # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 2.0  # 垂直方向の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 2.0  # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 1.2  # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0.0  # 凡例の端とグラフの端を合わせる

# Sample data
data1 = {
    "dataset": "MNIST",
    "x": [10, 100, 300, 1000, 10000],
    "y_kernelCNN": [32.72, 60.099999999999994, 72.96000000000001, 82.19, 91.93],
    "y_LeNet": [35.449999999999996, 62.49, 74.87, 84.50999999999999, 92.86],
    "y_HOG": [46.17, 69.89999999999999, 77.62, 85.15, 91.5],
}
data2 = {
    "dataset": "CIFAR10",
    "x": [10, 100, 300, 1000, 10000],
    "y_kernelCNN": [14.84, 18.54, 21.88, 23.84, 29.65],
    "y_LeNet": [15.67, 16.61, 23.94, 26.88, 34.489999999999995],
    "y_HOG": [17.86, 24.51, 31.380000000000003, 36.66, 44.72],
}
data3 = {
    "dataset": "KTH",
    "x": [10, 100, 300, 648],
    "y_kernelCNN": [9.25925925925926, 24.074074074074073, 35.18518518518518, 41.358024691358025],
    "y_LeNet": [30.246913580246915, 53.086419753086425, 65.4320987654321, 74.07407407407408],
    "y_HOG": [13.580246913580247, 30.864197530864196, 33.33333333333333, 43.20987654320987],
}
# Convert data to DataFrame
df = pd.DataFrame(data1)
dataset = df["dataset"][0]

fig = plt.figure(figsize=(10, 10))
fig_1 = fig.add_subplot(111)
# parameters
markersize = 15
markeredgewidth = 3.0
linewidth = 2.5  # Change the line width here
# Plot
## MNIST
df = pd.DataFrame(data1)
dataset = df["dataset"][0]
main_color = "r"
fig_1.plot(
    df["y_kernelCNN"],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    label=f"KernelCNN({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df["y_LeNet"],
    marker="s",
    markersize=markersize / 2,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    linestyle="--",
    label=f"LeNet({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df["y_HOG"],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor=main_color,
    markerfacecolor="w",
    color=main_color,
    linestyle=":",
    label=f"HOG({dataset})",
    linewidth=linewidth,
)

## FMNIST
df = pd.DataFrame(data2)
dataset = df["dataset"][0]
main_color = "b"
fig_1.plot(
    df["y_kernelCNN"],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    label=f"KernelCNN({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df["y_LeNet"],
    marker="s",
    markersize=markersize / 2,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    linestyle="--",
    label=f"LeNet({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df["y_HOG"],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor=main_color,
    markerfacecolor="w",
    color=main_color,
    linestyle=":",
    label=f"HOG({dataset})",
    linewidth=linewidth,
)

## CIFAR10
df = pd.DataFrame(data3)
dataset = df["dataset"][0]
main_color = "g"
fig_1.plot(
    df["y_kernelCNN"],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    label=f"KernelCNN({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df["y_LeNet"],
    marker="s",
    markersize=markersize / 2,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    linestyle="--",
    label=f"LeNet({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df["y_HOG"],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor=main_color,
    markerfacecolor="w",
    color=main_color,
    linestyle=":",
    label=f"HOG({dataset})",
    linewidth=linewidth,
)
# plot

# fig_1.plot(x, y, marker='.', markersize=10, markeredgewidth=1., markeredgecolor='k', color="y", label="test")
#
# fig_1.plot(x, y_2, marker='.', markersize=10, markeredgewidth=1., markeredgecolor='k', color="g", label="test_2")

fig_1.set_xlabel(r"Number of training samples $n$")
fig_1.set_ylabel(r"Accuracy(%)")
fig_1.set_ylim([0, 95])

x_labels = [10, 100, 300, 1000, 10000]
fig_1.set_xticks(range(len(x_labels)))
fig_1.set_xticklabels(x_labels)

fig_1.legend(ncol=3, bbox_to_anchor=(0.95, 0.05), loc="lower right")

# save
fig.savefig("graph.png", bbox_inches="tight", pad_inches=0.05)
