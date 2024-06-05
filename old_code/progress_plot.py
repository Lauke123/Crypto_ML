import numpy as np
import matplotlib.pyplot as plt

class ProgressPlot:
    def __init__(self):
        self.datacontainer = []

    def append_data(self, data:[float]):
        self.datacontainer.append(data)
    
    def generate_plot(self):
        for i, data in enumerate(self.datacontainer):
            y = np.array(data)
            x = np.arange(0, len(data))
            plt.plot(x,y, label=f"Pin{i}")
        leg = plt.legend(loc='lower right')
        plt.title("Learnigprogress of the different pins")
        plt.xlabel("Training Rounds")
        plt.ylabel("Accuracy")
        plt.savefig('progress_plot.png')


if __name__ == "__main__":
    progressplot = ProgressPlot()
    progressplot.append_data([1,2,15,4])
    progressplot.append_data([1,2,5,4,8,10])
    progressplot.append_data([1,2,3,4,10,12,3])
    progressplot.generate_plot()