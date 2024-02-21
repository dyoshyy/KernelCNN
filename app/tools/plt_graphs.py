import matplotlib.font_manager
del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 30 # 全体のフォントサイズが変更されます。
plt.rcParams['legend.fontsize'] = 20
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
plt.rcParams["legend.markerscale"] = 1.5 # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる

# Sample data
data = {
    'dataset' : "MNIST",
    'x': [1000, 10000, 30000, 60000],
    'y_kernelCNN': [94.72, 98.07, 98.83, 99.07],
    'y_LeNet': [95.8,98.73,98.94,99.03],
    'y_HOG': [94.42,96.69,97.58,97.98],
}

# Convert data to DataFrame
df = pd.DataFrame(data)
dataset = df['dataset'][0]

fig = plt.figure(figsize=(10,10))
fig_1 = fig.add_subplot(111)
#parameters
markersize = 15
markeredgewidth = 3.0
linewidth = 2.5  # Change the line width here
# Plot
fig_1.plot(df['y_kernelCNN'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor='none', color="r", label=f"KernelCNN({dataset})", linewidth=linewidth)
fig_1.plot(df['y_LeNet'], marker='s', markersize=markersize/2, markeredgewidth=markeredgewidth, markeredgecolor='none', color="r", linestyle='--', label=f"LeNet({dataset})", linewidth=linewidth)
fig_1.plot(df['y_HOG'], marker='.', markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor='r', markerfacecolor='w', color="r", linestyle=':', label=f"HOG({dataset})", linewidth=linewidth)


# plot

#fig_1.plot(x, y, marker='.', markersize=10, markeredgewidth=1., markeredgecolor='k', color="y", label="test")
#
#fig_1.plot(x, y_2, marker='.', markersize=10, markeredgewidth=1., markeredgecolor='k', color="g", label="test_2")

fig_1.set_xlabel(r"Number of training samples $n$")
fig_1.set_ylabel(r"Accuracy(%)")

fig_1.set_xticks(range(len(df['x'])))
fig_1.set_xticklabels(df['x'])

fig_1.legend(ncol=1, bbox_to_anchor=(0.95, 0.05), loc='lower right')

# save
fig.savefig('graph.png', bbox_inches="tight", pad_inches=0.05)