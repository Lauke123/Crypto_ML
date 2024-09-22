import os
from collections import Counter
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

    def plot_box_whisker_lugs(self, data: list[list], file_name: str):
        # Create boxplot
        plt.figure(figsize=(10, 6))  # Set the figure size
        boxplot = plt.boxplot(data, patch_artist=True)

        # Set x-axis labels
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], ["0-0", "0-1", "0-2", "0-3", "0-4", "0-5", "0-6", "1-2","1-3", "1-4", "1-5", "1-6","2-3", "2-4", "2-5", "2-6","3-4", "3-5", "3-6", "4-5","4-6", "5-6"])

        # Add labels and title
        plt.ylabel("Absolute Difference")
        plt.title("Prediction of Lug Pairs")

        # Calculate medians and annotate them on the plot
        medians = [item.get_ydata()[0] for item in boxplot['medians']]
        for i, median in enumerate(medians, start=1):
            plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        # Save the plot as an image file (PNG format)
        plt.savefig(self.storage_path + "/" + file_name)

        # Close the plot to prevent it from being displayed
        plt.close()
    
    def plot_lug_pair_accuracies(self, accuracies: list, file_name):

        average_value = np.mean(accuracies)
        plt.hist(accuracies, bins=28, color='blue',  edgecolor='black', alpha=0.7)
        plt.axvline(x=average_value, color='r', linestyle='--', label=f'Average: {average_value:.2f}')

        # Adding labels and title
        plt.xlabel('Accuracy Value')
        plt.ylabel('Frequency')
        plt.title('Frequency of Accuracy Values of Lug-Pair Prediction')

        # Add legend to indicate what the vertical line represents
        plt.legend()
        plt.savefig(self.storage_path + "/" + file_name)
        plt.close()
    
    def plot_avg_accuracy_distribution_pins_lugs(self, x_values, pin_accuracies, lug_accuracies, file_name):
        # Create the plot
        plt.plot(x_values, pin_accuracies, 'o-', color='red', label='Pins Accuracy')  # Red line with dots for pins
        plt.plot(x_values, lug_accuracies, 'o-', color='blue', label='Lugs Accuracy')  # Blue line with dots for lugs

        # Add labels and title
        plt.xlabel('Input Size')
        plt.ylabel('Accuracy')
        plt.title('Pins and Lugs Accuracy Comparison')

        # Add legend
        plt.legend()
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
