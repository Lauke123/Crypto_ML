import os

import numpy as np
from matplotlib import pyplot as plt


class PlotGenerator:
    def __init__(self, storage_path:str):
        self.storage_path = storage_path

    def generate_plot(self, accuracies: list, title:str, plot_name: str, bins:int=53):
        plt.clf()  # Clear the previous figure
        average_accuracy = np.mean(accuracies)
        plt.title(title)
        plt.hist(accuracies, bins=bins, color='lightgray',  edgecolor='black', alpha=0.7)
        plt.axvline(average_accuracy, color='red', linestyle='dashed', linewidth=1)
        plt.text(average_accuracy, plt.ylim()[1]/2, f'Average: {average_accuracy:.2f}%', rotation=90)

        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency")
        plt.savefig(self.storage_path + "/" + plot_name)
    
    def save_bar_chart(self, list1, list2, x_labels, name):
        """
        Creates a bar chart comparing two lists and saves it to the provided path.

        Parameters:
        - list1: First list of integers to compare.
        - list2: Second list of integers to compare.
        - x_labels: List of labels for the x-axis.
        - save_path: Path to save the bar chart image.
        """
        # Set the positions and width for the bars
        x = range(len(list1))  # positions for the bars
        bar_width = 0.35  # width of each bar

        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Compute the weighted average x position based on the y-values for list1
        total_weighted_x_list1 = sum(i * list1[i] for i in x)
        total_y_list1 = sum(list1)
        weighted_avg_x_list1 = total_weighted_x_list1 / total_y_list1 if total_y_list1 != 0 else 0

        # Compute the weighted average x position based on the y-values for list2
        total_weighted_x_list2 = sum(i * list2[i] for i in x)
        total_y_list2 = sum(list2)
        weighted_avg_x_list2 = total_weighted_x_list2 / total_y_list2 if total_y_list2 != 0 else 0

        # calculate the average accuracy of the pytorch and tensorflow prediction
        pytorch_avg = round(sum([list1[i] * float(x_labels[i]) for i in range(len(x_labels))])/sum(list1),2)
        tensorflow_avg = round(sum([list2[i] * float(x_labels[i]) for i in range(len(x_labels))])/sum(list2),2)

        # Add a dotted vertical line for the weighted average x value of list1
        plt.axvline(x=weighted_avg_x_list1 + bar_width / 2, color='b', linestyle='--', label=f'pytorch_avg={pytorch_avg}')

        # Add a dotted vertical line for the weighted average x value of list2
        plt.axvline(x=weighted_avg_x_list2 + bar_width / 2, color='g', linestyle='--', label=f'tensorflow_avg={tensorflow_avg}')



        # Plot the bars for list1 and list2 side by side
        plt.bar(x, list1, width=bar_width, label='pytorch', color='b', align='center')
        plt.bar([i + bar_width for i in x], list2, width=bar_width, label='tensorflow', color='g', align='center')

        # Add labels, title, and a legend
        plt.xlabel('Accuracy in %')
        plt.ylabel('Frequency')
        plt.title(f'Pytorch vs Tensorflow n={name}')
        plt.xticks([i + bar_width / 2 for i in x], x_labels)  # Set the x-ticks in the center
        plt.legend()

        # Save the plot to the specified file path
        plt.savefig(f"{self.storage_path}/{name}")

        # Close the plot to free up memory
        plt.close()


if __name__ == "__main__":
    test_path = os.path.join(os.getcwd(), "..")
    plotgenerator = PlotGenerator(test_path)
    test_list = [1,1,1,2,2,4,4,5,2,1,3,5]
    plotgenerator.generate_plot(test_list, "Test", "Testplot1")
    plotgenerator.generate_plot(test_list, "Test", "Testplot2")
