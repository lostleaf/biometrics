import numpy as np
import matplotlib.pyplot as plt
import math

def show(data):
    fig, ax = plt.subplots()
    x_lim, y_lim = data.shape
    x_ind = np.arange(x_lim)
    y_ind = np.arange(y_lim)
    x_val, y_val = np.meshgrid(x_ind, y_ind)
    for x, y, val in zip(x_val.flatten(), y_val.flatten(), data.T.flatten()):
        # if val > 0:
            color = 'black'
            if x > 2 and x < 12 and y > 2 and y < 7:
                color = 'red'
            ax.text(x, y, "%.0f" % val, va='center', ha='center', color=color)

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
    plt.show()


def main():
    data = np.zeros((15, 10), dtype=np.int32)
    data[1, 1] = 1
    data[1, 6] = 1
    data[4, 3] = 1
    data[4, 8] = 1
    data[8, 5] = 1
    data[14, 7] = 1
    # show(data)
    vote = np.zeros(data.shape)
    X, Y = np.where(data > 0)
    # X = [4]
    # Y = [3]
    for i in xrange(vote.shape[0]):
        for j in xrange(vote.shape[1]):
            for x, y in zip(X, Y):
                # dx1, dx2 = (i + 0.5 - x) ** 2, (i - 0.5 - x) ** 2
                # dy1, dy2 = (j + 0.5 - y) ** 2, (j - 0.5 - y) ** 2
                # dx_low, dx_high = min(dx1, dx2), max(dx1, dx2)
                # dy_low, dy_high = min(dy1, dy2), max(dy1, dy2)
                # dis_low, dis_high = dx_low + dy_low, dx_high + dy_high

                dis = math.sqrt((x - i) ** 2 + (y - j) ** 2)
                if abs(dis - 3) < 0.5 :
                    vote[i, j] += 1
                    # print i, j, dis_low, dis_high
                # vote[i, j] = dis

    show(vote)
if __name__ == "__main__":
    main()
