#
# In this file please complete the following task:
#
# Task 3 [6] Cross validation
# Evaluate your classifiers using the k-fold cross-validation technique
# covered in the lectures (use the training data only).
# Assume the number of folds is 100 (if the code takes too long to run, feel free to decrease it to as low as 10 folds).
# Output their average precisions, recalls, F-measures and accuracies. You need to implement the validation yourself.
import os
import Task_1_5


# In this task you are expected to perform cross-validation where f defines the number of folds to consider.
# "processed" holds the information from training data along with the following information: for each round in the
# cross validation technique, append a column stating the fold number in which a given image is,
# and the class it was assigned at this point (if it was in the testing fold)
# After everything is done, we add the average measures at the end. The writing to csv is done in a different function,
# and this is how I expect things to look:
# Example:
# path                  class                  round_1                        class_round_1                     round_2      class_round_2...
# <from training data>  <from training data>   in what fold it is in round 1  how it was classified in round 1
# ...
#
# avg_precision <value>
# avg_recall   <value>
# ...

def cross_evaluate_knn(training_data, k, sim_id, f):
    #
    rc = [[f'Round_{i}', f'Class_{i}'] for i in range(1, f + 1)]
    processed = [training_data[0] + [y for tp in rc for y in tp]]
    # Have fun!


    # Once folds are ready and you have the respective measures, please calculate the averages:
    avg_precision = float(0)
    avg_recall = float(0)
    avg_f_measure = float(0)
    avg_accuracy = float(0)
    # Have fun with the computations!
    # There are multiple ways to count average measures during cross-validation. For the purpose of this portfolio,
    # it's fine to just compute the values for each round and average them out in the usual way.


    # The measures are now added to the end:
    h = ['avg_precision', 'avg_recall','avg_f_measure', 'avg_accuracy']
    v = [avg_precision, avg_recall,avg_f_measure,avg_accuracy]
    r = [[h[i], v[i]] for i in range(len(h))]

    return processed + r


##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = Task_1_5.parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = Task_1_5.read_csv_file(opts['training_data'])
    print('Evaluating kNN')
    result = cross_evaluate_knn(training_data, opts['k'], opts['sim_id'], opts['f'])
    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{Task_1_5.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    Task_1_5.write_csv_file(out, result)


if __name__ == '__main__':
    main()