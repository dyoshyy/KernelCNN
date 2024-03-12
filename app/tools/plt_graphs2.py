import matplotlib.font_manager

del matplotlib.font_manager.weight_dict["roman"]
# matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"  # font familyの設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.size"] = 45  # 全体のフォントサイズが変更されます。
plt.rcParams["xtick.labelsize"] = 30  # 軸だけ変更されます。
plt.rcParams["ytick.labelsize"] = 30  # 軸だけ変更されます
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid
plt.rcParams["legend.fontsize"] = 15  # 凡例のフォントサイズ
plt.rcParams["legend.fancybox"] = False  # 丸角
plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 3  # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 1.5  # 垂直方向の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 1.0  # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 1.0  # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0.0  # 凡例の端とグラフの端を合わせる
plt.rcParams['figure.dpi'] = 300

line_styles = ['-', '--', '-.', ':']
main_color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]

# # baseline data
data1 = {
    "dataset": "MNIST",
    "x": [10, 100, 300, 648, 1000, 5000, 10000],
    "kernelCNN w/SLE": [44.37, 69.73, 75.31, 79.97, 85.03, 90.35, 91.25],
    "LeNet": [43.309999999999995, 70.59, 79.53, 84.06, 85.55, 91.7, 93.42],
    "HOG": [35.93, 73.49, 78.97, 83.54, 85.11, 90.38000000000001, 91.59],
}
data2 = {
    "dataset": "CIFAR10",
    "x": [10, 100, 300, 648, 1000, 5000, 10000],
    "kernelCNN w/LDA)": [17.39, 17.71, 22.43, 23.05, 24.959999999999997, 30.43, 33.800000000000004],
    "LeNet": [16.03, 18.64, 21.84, 23.45, 24.54, 32.16, 34.11],
    "HOG": [19.27, 25.019999999999996, 28.599999999999998, 29.98, 31.419999999999998, 38.61, 41.13],
}
data3 = {
    "dataset": "KTH",
    "x": [10, 100, 300, 648],
    "kernelCNN w/SLE": [51.85185185185185, 65.4320987654321, 70.9876, 74.07407407407408],
    "LeNet": [40.74074074074074, 58.0246913580247, 68.086419753086425, 76.5432098765432],
    "HOG": [36.41975308641975, 41.9753086419753, 54.93827160493827, 64.19753086419753],
}

# # embedding data
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
#     "SLE": [14.45, 18.77, 22.5, 25.69],
# }
# data3 = {
#     "dataset": "KTH",
#     "x": [10, 100, 300, 648],
#     "PCA": [13.580246913580247, 37.65432098765432, 40.74074074074074, 42.592592592592595],
#     "LDA": [46.913580246913575, 62.34567901234568, 67.28395061728395, 67.28395061728395],
#     "LE": [26.543209876543212, 40.74074074074074, 48.76543209876543, 48.76543209876543],
#     "SLE": [49.382716049382715, 64.19753086419753, 70.9876, 74.07407407407408],
# }

# # number of layers comaprison
# data1 = {
#     "dataset": "MNIST",
#     "x": [10, 100, 300, 648, 1000, 5000, 10000],
#     "LeNet": [43.309999999999995, 70.59, 79.53, 84.06, 85.55, 91.7, 93.42],
#     "3LayersCNN": [32.54, 70.7, 79.94, 84.5, 85.64, 91.81, 93.04],
# }
# data2 = {
#     "dataset": "CIFAR10",
#     "x": [10, 100, 300, 648, 1000, 5000, 10000],
#     "LeNet": [16.03, 18.64, 21.84, 23.45, 24.54, 32.16, 34.11],
#     "3LayersCNN": [14.030000000000001, 18.88, 21.19, 24.0, 26.26, 33.410000000000004, 34.88],
# }
# data3 = {
#     "dataset": "KTH",
#     "x": [10, 100, 300, 648],
#     "LeNet": [40.74074074074074, 58.0246913580247, 68.086419753086425, 76.5432098765432],
#     "3LayersCNN": [34.5679012345679, 53.70370370370371, 61.111111111111114, 67.90123456790124],
# }

# # CNN vs LE vs SLE
# data1 = {
#     "dataset": "MNIST",
#     "x": [10, 100, 300, 648, 1000, 5000, 10000],
#     "3LayersCNN": [32.54, 70.7, 79.94, 84.5, 87.64, 91.81, 93.04],
#     "KernelCNN w/SLE": [44.37, 69.73, 75.31, 79.97, 85.03, 90.35, 91.25],
#     "KernelCNN w/LE": [38.269999999999996, 69.02000000000001, 75.3, 79.7, 84.03, 88.71, 90.55],
# }
# data2 = {
#     "dataset": "CIFAR10",
#     "x": [10, 100, 300, 648, 1000, 5000, 10000],
#     "3LayersCNN": [14.030000000000001, 18.88, 21.19, 24.0, 26.26, 33.410000000000004, 34.88],
#     "KernelCNN w/SLE": [18.05, 19.74, 22.18, 24.73, 25.0, 29.39, 33.35],
#     "KernelCNN w/LE": [16.5, 17.39, 18.96, 20.0, 22.38, 26.32, 30.11],
# }
# data3 = {
#     "dataset": "KTH",
#     "x": [10, 100, 300, 648],
#     "3LayersCNN": [34.5679012345679, 53.70370370370371, 61.111111111111114, 67.90123456790124],
#     "KernelCNN w/SLE": [49.382716049382715, 64.19753086419753, 70.9876, 74.07407407407408],
#     "KernelCNN w/LE": [26.543209876543212, 40.74074074074074, 48.76543209876543, 48.76543209876543],
# }

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)

dataFrames = [df1, df2, df3]

# Convert data to DataFrame
df = pd.DataFrame(data1)
dataset = df["dataset"][0]

fig = plt.figure(figsize=(15, 15))
fig_1 = fig.add_subplot(111)
# parameters
markersize = 10
markeredgewidth = 3.0
linewidth = 2.5  # Change the line width here
# Plot
## MNIST
# df = pd.DataFrame(data1)
# dataset = df["dataset"][0]
# print(df.iloc[:, 2])
for j in range(len(dataFrames)):
    df = dataFrames[j]
    for i in range(df.shape[1]-2):
        dataset = df["dataset"][0]
        fig_1.plot(
            df.iloc[:, i+2],
            marker="o",
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            markeredgecolor="none",
            color=main_color_list[j],
            linestyle = line_styles[i],
            label=f"{df.columns[i+2]}({dataset})",
            linewidth=linewidth,
    )

fig_1.set_xlabel(r"Number of training samples $n$")
fig_1.set_ylabel(r"Accuracy(%)")
fig_1.set_ylim([0, 95])

fig_1.set_xticks(range(len(df1["x"])))
fig_1.set_xticklabels(df1["x"])

fig_1.legend(ncol=3, bbox_to_anchor=(0.975, 0.025), loc="lower right")

# save
fig.savefig("graph.png", bbox_inches="tight", pad_inches=0.05)
# fig.savefig("graph.eps", bbox_inches="tight", pad_inches=0.05)

