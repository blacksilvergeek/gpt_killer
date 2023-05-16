import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors


def stacked_bar_plot(labels, mean_1, mean2, err_1, err_2, label_1, label_2):
    plt.style.use('fivethirtyeight')

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, mean_1, width, label=label_1, yerr=err_1)
    rects2 = ax.bar(x + width / 2, mean2, width, label=label_2, yerr=err_2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Classification error')
    # ax.set_title('Error by author and network type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune='both'))

    plt.show()


def box_plot_with_dist(labels, data, title, y_label='Classification error'):
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data, labels=labels)
    cmap = mcolors.ListedColormap(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for i in range(len(labels)):
        y = data[i]
        x = [i + 1] * len(y)
        ax.scatter(x, y, cmap=cmap, alpha=0.4)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    plt.show()
