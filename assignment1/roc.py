import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.array([0, 0, 100])
    y = np.array([0, 100, 100])
    plt.plot(x,y,linewidth=5)
    plt.ylim([0, 110])
    plt.xlabel("FNMR(%)")
    plt.ylabel("TMR(%)")
    plt.title("ROC Curve")
    plt.show()

if __name__ == "__main__":
    main()
