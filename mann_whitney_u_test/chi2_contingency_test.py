import csv
from scipy.stats import chi2_contingency
import os 

import matplotlib.pyplot as plt


def save_bar_chart(list1, list2, x_labels, name):
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
    plt.savefig(f"chi_test_{name}")

    # Close the plot to free up memory
    plt.close()

# test if there is a significant difference between the distribution of accuracies of
# tensorflow and pytorch implementation

# test for input size 52
# Read CSV files
print(os.getcwd())

    
def chi2_test(pytorch_prediction_path, tensorflow_prediction_path, name):
    # perform chi2_contingency test
    chi_pytorch = {}
    chi_tensorflow = {}


    with open(pytorch_prediction_path) as fp:
        pytorch_reader = csv.reader(fp)
        pytorch_accuracies = [float(row[0]) for row in pytorch_reader]
        for accuracy in pytorch_accuracies:
            if accuracy not in chi_pytorch.keys():
                chi_pytorch[accuracy] = 1
            else:
                chi_pytorch[accuracy] += 1

    with open(tensorflow_prediction_path) as fp:
        tensorflow_reader = csv.reader(fp)
        tensorflow_accuracies = [float(row[0]) for row in tensorflow_reader]
        for accuracy in tensorflow_accuracies:
            if accuracy not in chi_tensorflow.keys():
                chi_tensorflow[accuracy] = 1
            else:
                chi_tensorflow[accuracy] += 1

    all_keys = set(chi_pytorch.keys()).union(chi_tensorflow.keys())

    pytorch_values = []
    tensorflow_values = []
    key_values = []

    for key in sorted(all_keys):
        key_values.append(key)
        pytorch_values.append(chi_pytorch.get(key, 0))
        tensorflow_values.append(chi_tensorflow.get(key, 0))

    # save plot of the two frequency distributions
    key_values = [ '%.0f' % elem for elem in key_values ]
    save_bar_chart(pytorch_values, tensorflow_values, key_values, name)

    output = [pytorch_values, tensorflow_values]
    output = chi2_contingency(output)
    return output

result_52 = chi2_test("mann_whitney_u_test/pytorch_prediction_accuracies_52.csv", "mann_whitney_u_test/tensorflow_prediction_accuracies_52.csv", "52")
print(result_52)

result_104 = chi2_test("mann_whitney_u_test/pytorch_prediction_accuracies_104.csv", "mann_whitney_u_test/tensorflow_prediction_accuracies_104.csv", "104")
print(result_104)

result_200 = chi2_test("mann_whitney_u_test/pytorch_prediction_accuracies_200.csv", "mann_whitney_u_test/tensorflow_prediction_accuracies_200.csv", "200")
print(result_200)