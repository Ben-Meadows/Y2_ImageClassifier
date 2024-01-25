#
# In this file please complete the following task:
#
# Task 2 [2] Basic evaluation
# Evaluate your classifiers. On your own, implement methods that will output
# precision, recall, F-measure, and accuracy of your classifiers.
#
# You are expected to rely on solutions From file Task_1_5.py here!

import Task_1_5


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier
def evaluate_knn(training_data, k, sim_id, data_to_classify):
    precision = float(0)
    recall = float(0)
    f_measure = float(0)
    accuracy = float(0)
    # Have fun with the computations!

    # once ready, we return the values
    return precision, recall, f_measure, accuracy


##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Task_1_5.parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = Task_1_5.read_csv_file(opts['training_data'])
    data_to_classify = Task_1_5.read_csv_file(opts['data_to_classify'])
    print('Evaluating kNN')
    result = evaluate_knn(training_data, opts['k'], opts['sim_id'], data_to_classify)
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()