import matplotlib.font_manager

del matplotlib.font_manager.weight_dict["roman"]
# matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"  # font familyの設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.size"] = 30  # 全体のフォントサイズが変更されます。
plt.rcParams["legend.fontsize"] = 10  # 凡例のフォントサイズ
plt.rcParams["xtick.labelsize"] = 25  # 軸だけ変更されます。
plt.rcParams["ytick.labelsize"] = 25  # 軸だけ変更されます
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid
plt.rcParams["legend.fancybox"] = False  # 丸角
plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 2.5  # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 1.5  # 垂直方向の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 1.5  # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 1.0  # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0.0  # 凡例の端とグラフの端を合わせる

# # baseline data
data1 = {
    "dataset": "MNIST",
    "x": [10, 100, 300, 648],
    "kernelCNN": [34.12, 61.47, 72.09, 79.46],
    "LeNet": [36.84, 63.06, 74.38, 82.15],
    "HOG": [44.529999999999994, 66.67999999999999, 76.67, 80.82000000000001],
}
data2 = {
    "dataset": "CIFAR10",
    "x": [10, 100, 300, 648],
    "kernelCNN": [15.28, 18.360000000000003, 22.400000000000002, 23.78],
    "LeNet": [15.740000000000002, 16.7, 24.44, 24.89],
    "HOG": [16.72, 21.5, 28.01, 31.85],
}
data3 = {
    "dataset": "KTH",
    "x": [10, 100, 300, 648],
    "kernelCNN": [6.172839506172839, 40.123456790123456, 43.82716049382716, 59.876543209876544],
    "LeNet": [16.666666666666664, 48.148148148148145, 58.64197530864198, 63.580246913580254],
    "HOG": [13.580246913580247, 45.06172839506173, 62.96296296296296, 70.98765432098766],
}

# # baseline data
# data1 = {
#     "dataset": "MNIST",
#     "x": [10, 100, 300, 648],
#     "PCA": [35.82, 68.04, 77.23, 81.2],
#     "LDA": [45.03, 66.67999999999999, 77.77, 82.3],
#     "LE": [41.49, 65.94, 77.07000000000001, 82.28],
#     "SLE": [44.37, 64.73, 75.31, 79.97],
# }
# data2 = {
#     "dataset": "CIFAR10",
#     "x": [10, 100, 300, 648],
#     "PCA": [16.18, 17.22, 21.15, 23.7],
#     "LDA": [14.62, 19.32, 21.7, 25.22],
#     "LE": [15.75, 15.72, 21.16, 23.330000000000002],
#     "SLE": [14.45, 18.77, 10.0, 25.69],
# }
# data3 = {
#     "dataset": "KTH",
#     "x": [10, 100, 300, 648],
#     "PCA": [13.580246913580247, 37.65432098765432, 40.74074074074074, 42.592592592592595],
#     "LDA": [46.913580246913575, 62.34567901234568, 67.28395061728395, 67.28395061728395],
#     "LE": [26.543209876543212, 40.74074074074074, 48.76543209876543, 48.76543209876543],
#     "SLE": [49.382716049382715, 64.19753086419753, 9.876543209876543, 74.07407407407408],
# }

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
# print(df.iloc[:, 2])
main_color = "r"
fig_1.plot(
    df.iloc[:, 2],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    label=f"{df.columns[2]}({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df.iloc[:, 3],
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
    df.iloc[:, 4],
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
    df.iloc[:, 2],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    label=f"KernelCNN w/LE({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df.iloc[:, 3],
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
    df.iloc[:, 4],
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
    df.iloc[:, 2],
    marker=".",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    markeredgecolor="none",
    color=main_color,
    label=f"KernelCNN w/LE({dataset})",
    linewidth=linewidth,
)
fig_1.plot(
    df.iloc[:, 3],
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
    df.iloc[:, 4],
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
fig_1.set_ylim([0, 85])

fig_1.set_xticks(range(len(df["x"])))
fig_1.set_xticklabels(df["x"])

fig_1.legend(ncol=3, bbox_to_anchor=(0.975, 0.025), loc="lower right")

# save
fig.savefig("graph.png", bbox_inches="tight", pad_inches=0.05)
