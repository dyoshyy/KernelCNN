import matplotlib.font_manager
del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 30 # 全体のフォントサイズが変更されます。
plt.rcParams['legend.fontsize'] = 10 # 凡例のフォントサイズ
plt.rcParams['xtick.labelsize'] = 25 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 25 # 軸だけ変更されます
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid
plt.rcParams["legend.fancybox"] = False # 丸角
plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1.2 # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 2. # 垂直方向の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 2. # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 1.2 # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる

# Sample data
data1 = {
    'dataset' : "MNIST",
    'x': [1000, 10000, 30000, 60000],
    'y_kernelCNN': [88.31, 89.88, 94.62, 95.26],
    'y_LeNet': [89.76, 94.54, 98.68, 98.82],
    'y_HOG': [92.72, 95.8, 97.01, 97.21],
}
data2 = {
    'dataset' : "FMNIST",
    'x': [1000, 10000, 30000, 60000],
    'y_kernelCNN': [80.80, 86.25, 88.07, 88.81],
    'y_LeNet': [80.71, 86.19, 88.52, 89.03],
    'y_HOG': [79.91, 84.23,	85.16, 85.86],
}
data3 = {
    'dataset' : "CIFAR10",
    'x': [1000, 10000, 30000, 60000],
    'y_kernelCNN': [39.44, 47.85, 50.58, 52.56],
    'y_LeNet': [39.11, 51.83, 57.64, 60.54],
    'y_HOG': [10, 10, 10, 10],
}

# Convert data to DataFrame
df = pd.DataFrame(data1)
dataset = df['dataset'][0]

fig = plt.figure(figsize=(10,10))
fig_1 = fig.add_subplot(111)
#parameters
markersize = 15
markeredgewidth = 3.0
linewidth = 2.5  # Change the line width here
# Plot
## MNIST
df = pd.DataFrame(data1)
dataset = df['dataset'][0]
main_color = "r"
fig_1.plot(df['y_kernelCNN'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor='none', color=main_color, label=f"KernelCNN({dataset})", linewidth=linewidth)
fig_1.plot(df['y_LeNet'], marker='s', markersize=markersize/2, markeredgewidth=markeredgewidth, markeredgecolor='none', color=main_color, linestyle='--', label=f"LeNet({dataset})", linewidth=linewidth)
fig_1.plot(df['y_HOG'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor=main_color, markerfacecolor='w', color=main_color, linestyle=':', label=f"HOG({dataset})", linewidth=linewidth)

## FMNIST
df = pd.DataFrame(data2)
dataset = df['dataset'][0]
main_color = "b"
fig_1.plot(df['y_kernelCNN'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor='none', color=main_color, label=f"KernelCNN({dataset})", linewidth=linewidth)
fig_1.plot(df['y_LeNet'], marker='s', markersize=markersize/2, markeredgewidth=markeredgewidth, markeredgecolor='none', color=main_color, linestyle='--', label=f"LeNet({dataset})", linewidth=linewidth)
fig_1.plot(df['y_HOG'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor=main_color, markerfacecolor='w', color=main_color, linestyle=':', label=f"HOG({dataset})", linewidth=linewidth)

## CIFAR10
df = pd.DataFrame(data3)
dataset = df['dataset'][0]
main_color = "g"
fig_1.plot(df['y_kernelCNN'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor='none', color=main_color, label=f"KernelCNN({dataset})", linewidth=linewidth)
fig_1.plot(df['y_LeNet'], marker='s', markersize=markersize/2, markeredgewidth=markeredgewidth, markeredgecolor='none', color=main_color, linestyle='--', label=f"LeNet({dataset})", linewidth=linewidth)
fig_1.plot(df['y_HOG'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor=main_color, markerfacecolor='w', color=main_color, linestyle=':', label=f"HOG({dataset})", linewidth=linewidth)
# plot

#fig_1.plot(x, y, marker='.', markersize=10, markeredgewidth=1., markeredgecolor='k', color="y", label="test")
#
#fig_1.plot(x, y_2, marker='.', markersize=10, markeredgewidth=1., markeredgecolor='k', color="g", label="test_2")

fig_1.set_xlabel(r"Number of training samples $n$")
fig_1.set_ylabel(r"Accuracy(%)")

fig_1.set_xticks(range(len(df['x'])))
fig_1.set_xticklabels(df['x'])

fig_1.legend(ncol=3, bbox_to_anchor=(0.95, 0.05), loc='lower right')

# save
fig.savefig('graph.png', bbox_inches="tight", pad_inches=0.05)