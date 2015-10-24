import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as pat


def adjust_axis(x_lim, y_lim, ax):
    ax.set_xlim(0, x_lim - 0.5)
    ax.set_ylim(0, y_lim - 0.5)
    ax.invert_yaxis()

    xlocations = np.arange(x_lim)
    xminorlocations = xlocations - 0.5
    plt.xticks(xlocations, [str(x+1) for x in np.arange(x_lim)])
    plt.gca().xaxis.set_ticks_position('none')
    ax.set_xticks(xminorlocations, minor=True)

    ylocations = np.arange(y_lim)
    yminorlocations = ylocations - 0.5
    plt.yticks(ylocations, [str(x+1) for x in np.arange(y_lim)])
    plt.gca().yaxis.set_ticks_position('none')
    ax.set_yticks(yminorlocations, minor=True)

    plt.grid(True, which='minor', linestyle='-')


def show_data(data):
    fig, ax = plt.subplots()
    x_lim, y_lim = data.shape
    x_ind = np.arange(x_lim)
    y_ind = np.arange(y_lim)
    x_val, y_val = np.meshgrid(x_ind, y_ind)
    for x, y, val in zip(x_val.flatten(), y_val.flatten(), data.T.flatten()):
        color = 'black'
        if x > 2 and x < 12 and y > 2 and y < 7:
            color = 'red'
        ax.text(x, y, "%.0f" % val, va='center', ha='center', color=color)

    adjust_axis(x_lim, y_lim, ax)
    plt.show()


def show_circle(data, centers):
    for i, (cx, cy) in enumerate(centers):
        ax = plt.subplot(321 + i)
        x_lim, y_lim = data.shape
        x_ind = np.arange(x_lim)
        y_ind = np.arange(y_lim)
        x_val, y_val = np.meshgrid(x_ind, y_ind)
        for x, y, val in zip(x_val.flatten(), y_val.flatten(), data.T.flatten()):
            on_circle = abs(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - 3) < 0.5
            if val > 0:
                color = 'blue'
                if on_circle:
                    color = 'black'
                patch = pat.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color)
                ax.add_patch(patch)
            elif on_circle:
                patch = pat.Rectangle((x - 0.5, y - 0.5), 1, 1, color='green')
                ax.add_patch(patch)

        adjust_axis(x_lim, y_lim, ax)
    plt.show()


def main():
    data = np.zeros((15, 10), dtype=np.int32)
    data[1, 1] = 1
    data[1, 6] = 1
    data[4, 3] = 1
    data[4, 8] = 1
    data[8, 5] = 1
    data[14, 7] = 1

    vote = np.zeros(data.shape)
    X, Y = np.where(data > 0)

    for i in xrange(vote.shape[0]):
        for j in xrange(vote.shape[1]):
            for x, y in zip(X, Y):
                dis = math.sqrt((x - i) ** 2 + (y - j) ** 2)
                if abs(dis - 3) < 0.5:
                    vote[i, j] += 1

    max_vote = np.max(vote)
    X, Y = np.where(vote == max_vote)
    centers = [(x, y)
               for x, y in zip(X, Y) if x > 2 and x < 12 and y > 2 and y < 7]
    show_data(vote)

    # show_circle(data, centers)

if __name__ == "__main__":
    main()
