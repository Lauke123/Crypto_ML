import os

import numpy as np
from matplotlib import pyplot as plt


class PlotGenerator:
    def __init__(self, storage_path:str):
        self.storage_path = storage_path

    def generate_plot(self, accuracies: list, title:str, plot_name: str):
        plt.clf()  # Clear the previous figure
        average_accuracy = np.mean(accuracies)
        plt.title(title)
        plt.hist(accuracies, bins=66, color='wheat',  edgecolor='black', alpha=0.7)
        plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
        plt.text(average_accuracy, plt.ylim()[1]/2, f'Average: {average_accuracy:.2f}%', rotation=90)

        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency")
        plt.savefig(self.storage_path + "/" + plot_name)

    def plot_box_whisker_lugs(self, data:list[list], file_name):

        # Create boxplot
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.boxplot(data)

        # Set x-axis labels
        plt.xticks([1, 2, 3, 4, 5, 6, 7], ["No Wheel", "Wheel1", "Wheel2", "Wheel3", "Wheel4", "Wheel5", "Wheel6"])

        # Add labels and title
        plt.ylabel("Absolute Difference")
        plt.title("Prediction of Lug Distribution")

        # Save the plot as an image file (PNG format)
        plt.savefig(self.storage_path + "/" + file_name)

        # Close the plot to prevent it from being displayed
        plt.close()

if __name__ == "__main__":
    test_path = os.path.join(os.getcwd(), "..")
    plotgenerator = PlotGenerator(test_path)
    test_list = [1,1,1,2,2,4,4,5,2,1,3,5]
    plotgenerator.generate_plot(test_list, "Test", "Testplot1")
    plotgenerator.generate_plot(test_list, "Test", "Testplot2")
