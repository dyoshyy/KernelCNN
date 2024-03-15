import matplotlib.font_manager
import numpy as np

del matplotlib.font_manager.weight_dict["roman"]
# matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import pandas as pd
import json

plt.rcParams["font.family"] = "Times New Roman"  # font familyの設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.size"] = 55  # 全体のフォントサイズが変更されます。
plt.rcParams["xtick.labelsize"] = 40  # 軸だけ変更されます。
plt.rcParams["ytick.labelsize"] = 40  # 軸だけ変更されます
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid
plt.rcParams["legend.fontsize"] = 22  # 凡例のフォントサイズ
plt.rcParams["legend.fancybox"] = False  # 丸角
plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 3  # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 1.5  # 垂直方向の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 1.0  # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 0.8  # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0.0  # 凡例の端とグラフの端を合わせる
plt.rcParams['figure.dpi'] = 300

line_styles = ['-', ':', '--', '-.']
main_color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]

# data = "baseline" # 1 convolution
data = "deepening_comparison"
# data = "embedding_comparison"

# classifier = "1NN"
classifier = "SVM"

with open('data/' + classifier + '/' + data + '.json', 'r') as f:
    data = json.load(f)

dataFrames = []
for i in range(len(data)):
    dataFrames.append(pd.DataFrame(data[i]))

for i, feature in enumerate(["Embedded", "Deep"]):
    
    # data
    labels = ["MNIST", "CIFAR10", "KTH_TIP"]
    accuracy_1 = []
    accuracy_2 = []

    for df in dataFrames:
        print("df:", df)
        accuracy_1.append(df.iloc[-1, i*2+2]) # n=50,000
        accuracy_2.append(df.iloc[-1, i*2+3])

    print(accuracy_1)
    print(accuracy_2)

    # parameters
    width = 0.3

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    # Plot
    left = np.arange(len(labels))
    ax.set_xlabel(r"Datasets")
    ax.set_ylabel(r"Accuracy(%)")
    ax.bar(left , accuracy_1, width, color="c", align="center")
    ax.bar(left + width, accuracy_2, width, color="b", align="center")

    ax.set_xticks(left + width/2, labels)
    ax.set_ylim([40, 100])
    ax.grid(False)
    fig.savefig(f"bar_graph_{feature}.png", bbox_inches="tight", pad_inches=0.05)



