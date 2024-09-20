import csv
from scipy.stats import mannwhitneyu, ttest_ind
import os 
# test if there is a significant difference between the distribution of accuracies of
# tensorflow and pytorch implementation

# test for input size 52
# Read CSV files
print(os.getcwd())
with open("mann_whitney_u_test/pytorch_prediction_accuracies_52.csv") as fp:
    pytorch_reader_52 = csv.reader(fp)
    pytorch_52_accuracies = [float(row[0]) for row in pytorch_reader_52]
with open("mann_whitney_u_test/tensorflow_prediction_accuracies_52.csv") as fp:
    tensorflow_reader_52 = csv.reader(fp)
    tensorflow_52_accuracies = [float(row[0]) for row in tensorflow_reader_52]
result_52 = mannwhitneyu(pytorch_52_accuracies, tensorflow_52_accuracies)
result_52_ttest_ind = ttest_ind(pytorch_52_accuracies, tensorflow_52_accuracies)
print(f"mann_whitney_u: {result_52}")
print(f"t_test: {result_52_ttest_ind}")

with open("mann_whitney_u_test/pytorch_prediction_accuracies_104.csv") as fp:
    pytorch_reader_104 = csv.reader(fp)
    pytorch_104_accuracies = [float(row[0]) for row in pytorch_reader_104]
with open("mann_whitney_u_test/tensorflow_prediction_accuracies_104.csv") as fp:
    tensorflow_reader_104 = csv.reader(fp)
    tensorflow_104_accuracies = [float(row[0]) for row in tensorflow_reader_104]
result_104 = mannwhitneyu(pytorch_104_accuracies, tensorflow_104_accuracies)
result_104_ttest_ind = ttest_ind(pytorch_104_accuracies, tensorflow_104_accuracies)
print(f"mann_whitney_u: {result_104}")
print(f"t_test: {result_104_ttest_ind}")

with open("mann_whitney_u_test/pytorch_prediction_accuracies_200.csv") as fp:
    pytorch_reader_200 = csv.reader(fp)
    pytorch_200_accuracies = [float(row[0]) for row in pytorch_reader_200]
with open("mann_whitney_u_test/tensorflow_prediction_accuracies_200.csv") as fp:
    tensorflow_reader_200 = csv.reader(fp)
    tensorflow_200_accuracies = [float(row[0]) for row in tensorflow_reader_200]
result_200 = mannwhitneyu(pytorch_200_accuracies, tensorflow_200_accuracies)
result_200_ttest_ind = ttest_ind(pytorch_200_accuracies, tensorflow_200_accuracies)
print(f"mann_whitney_u: {result_200}")
print(f"t_test: {result_200_ttest_ind}")