import matplotlib.font_manager

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

data = "baseline" # 1 convolution

# data = "embedding_comparison"
# data = "number_of_layers_comparison"

# classifier = "1NN"
classifier = "SVM"

with open('data/' + classifier + '/' + data + '.json', 'r') as f:
    data = json.load(f)

dataFrames = []
for i in range(len(data)):
    dataFrames.append(pd.DataFrame(data[i]))
    
# parameters
markersize = 10
markeredgewidth = 3.0
linewidth = 2.5  

# Plot
for j in range(len(dataFrames)):
    fig = plt.figure(figsize=(15, 15))
    # fig_1 = fig.add_subplot(111)
    ax = fig.add_subplot(111)
    df = dataFrames[j]
    for i in range(df.shape[1]-2):
        dataset = df["dataset"][0]
        ax.plot(
            df.iloc[:, i+2],
            marker="o",
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            markeredgecolor="none",
            color=main_color_list[j],
            linestyle = line_styles[i],
            label=f"{df.columns[i+2]}",
            linewidth=linewidth,
    )
    ax.set_xlabel(r"Number of training samples $n$")
    ax.set_ylabel(r"Accuracy(%)")
    # fig_1.set_ylim([0, 100])

    ax.set_xticks(range(len(df["x"])))
    ax.set_xticklabels(df["x"])
    ax.legend(ncol=3, bbox_to_anchor=(0.975, 0.025), loc="lower right")

    # save
    fig.savefig(f"{dataset}.png", bbox_inches="tight", pad_inches=0.05)



