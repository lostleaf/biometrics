import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.array([0.1] * 5 + [0.5] * 20)
    N = 5
    a = (5, 0, 0, 0, 0)
    b = (0, 0, 20, 0, 0)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind + 0.17, a, width, color='r')

    rects2 = ax.bar(ind + 0.17, b, width, color='y')

# add some text for labels, title and axes ticks
    plt.ylim([0,25])
    ax.set_ylabel('Number')
    ax.set_xlabel('Score')
    ax.set_title('Imposter and Genuine Distributions')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(np.arange(0.1, 1.0, 0.2))

    ax.legend( (rects1[0], rects2[0]), ('Genuine', 'Impostor') )

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)

    plt.show()

if __name__ == "__main__":
    main()
