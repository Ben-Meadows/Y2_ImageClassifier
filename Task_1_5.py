import argparse
import csv
import os
import pandas as pd
import numpy as np
from PIL import Image
from sewar.full_ref import mse, rmse

#
# In this file please complete the following tasks:
#
# Task 1 [10] My first not-so-pretty image classifier
# By using the kNN approach and three similarity measures, build image classifiers.
# You need to implement the kNN approach yourself, however, you can use libraries for any similarity measures.
# You can assume that k=100 (if the code takes too long to run, feel free to decrease it to as low as k=10).
# You are allowed to use libraries to read and write to files, and to perform image transformations if necessary.

# Task 5 [6] Similarities
# Independent inquiry time! In Task 1, you were allowed to use libraries for image similarity measures.
# Pick two of the three measures you have used and implement them yourself!


# Please replace with your student id!!!
student_id = '2064936'

# This is the classification scheme you should use for kNN
classification_scheme = ['Female','Male','Primate','Rodent','Food']


# In this function I expect you to implement the kNN classifier. You are free to define any number of helper functions
# you need for this!
#
# INPUT: training_data      : a list of lists that was read from the training data csv (see parse_arguments function)
#        k                  : the value of k neighbours
#        sim_id             : value from 1 to 5 which says what similarity should be used;
#                             values from 1, 2 and 3 denote similarities from Task 1 that can be called from libraries
#                             values from 4 and 5 denote similarities from Task 5 that you implement yourself
#        data_to_classify   : a list of lists that was read from the data to classify csv;
#                             this data is NOT be used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
# OUTPUT: processed         : a list of lists which expands the data_to_classify with the results on how your
#                             classifier has classified a given image

def kNN(training_data, k, sim_id, data_to_classify):

    processed = [data_to_classify[0] + [student_id]]
    # Reading Test and Training Images, into an numpy array to calculate similarities
    # Using [1:] as the first row has the Name of the Columns
    # Making arrays with the information of the images from Testing and Training Images
    count = 0
    precision = 0
    for imagepth in data_to_classify[1:]:
        # have to resize image so they all have the same size
        #opens up image and rizes it to 64 pixles by 64 had to do this cause this size produced a resonable output time with an okay accracy
        #can be changed to higher pixels for more acrucy but my laptop does not have much proecessing power
        #stores in a numpy array every rgb value of every pixel in the new rezised image these can be used to find similiraties
        #this is a loist of lists where there are 64 arrays containing all the rgb values of every line of pixels
        image_studentTest = np.array(Image.open(imagepth[0]).resize((64, 64)))
        start_class=(imagepth[1])
        print(start_class)

        Distance2= {}
        for trainpth in training_data[1:]:
            #using PIL package here toi open and edit image
            image_studentTrain = np.array(Image.open(trainpth[0]).resize((64, 64)))
            # Three Similarity Metrics (SIM)
            if sim_id == 1:
                 #Mean Squared Error (MSE)]
                distance = mse(image_studentTest, image_studentTrain)
            elif sim_id == 2:
                #Euclidean Distance turn the arrays into vectors and can take away from each other
                #then normalize the new vector which give us an int which is the distance
                distance = np.linalg.norm(image_studentTest - image_studentTrain)
            else:
                # Root Mean Squared Error (RMSE)
                distance = rmse(image_studentTest, image_studentTrain)

            Distance2[distance] = trainpth[1]
        #how classication is chosen
        #reorder dict to be from lowest to highest
        Distance3 = dict(sorted(Distance2.items()))
        #initate all the values
        female=0
        male=0
        primate=0
        rodent=0
        food=0
        for idx, (value) in enumerate(Distance3.values()):
            if idx == k: break

            if value == ('Female'):
                female +=1
            if value == ('Male'):
                male+=1
            if value ==('Primate'):
                primate+=1
            if value ==('Rodent'):
                rodent+=1
            if value ==('Food'):
                food+=1
        #add the valuesinto a list so i can find the biggest value so to find the nearest neighbour
        find_nearestneighbour= []
        find_nearestneighbour.append(female)
        find_nearestneighbour.append(male)
        find_nearestneighbour.append(primate)
        find_nearestneighbour.append(rodent)
        find_nearestneighbour.append(food)
        #finds the max value in this list
        x= max(find_nearestneighbour)
        #finds the postion in the array casue 0=female 1=male 2=primate 3=rodent 4=food
        p=find_nearestneighbour.index(x)

        if p == 0:
            Classification = 'Female'
        if p == 1:
            Classification = 'Male'
        if p == 2:
            Classification = 'Primate'
        if p == 3:
            Classification = 'Rodent'
        if p == 4:
            Classification = 'Food'
        count = count + 1
        if start_class == Classification:
            precision +=1
        print(count)
        print('The guessed Class is: '+Classification)
        print('')
        # Compiling everything for it to be printed into a csv file
        processed.append([imagepth[0], imagepth[1], Classification])
    print((precision/count)*100)
    return processed


##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################

# This function reads the necessary arguments (see parse_arguments function), and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.
def main():
    opts = parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = read_csv_file(opts['training_data'])
    data_to_classify = read_csv_file(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    result = kNN(training_data, opts['k'], opts['sim_id'], data_to_classify)
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        write_csv_file(out, result)


# Straightforward function to read the data contained in the file "filename"
def read_csv_file(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# Straightforward function to write the data contained in "lines" to a file "filename"
def write_csv_file(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#                         (needed in Tasks 1, 2, 3 and 5)
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -sim_id         : value from 1 to 5 which says what similarity should be used;
#                         values from 1, 2 and 3 denote similarities from Task 1 that can be called from libraries
#                         values from 4 and 5 denote similarities from Task 5 that you implement yourself
#                         (needed in Tasks 1, 2, 3 and 5)
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#                         (needed in Tasks 1, 3 and 5)
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#                         (needed in Tasks 1, 2, 3 and 5)
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#                         (needed in Tasks 1, 2, 3 and 5)
#
def parse_arguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-s', '--sim_id', nargs='?', type=int)
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('training_data')
    parser.add_argument('data_to_classify')
    params = parser.parse_args()

    if params.sim_id < 0 or params.sim_id > 5:
        print('Argument sim_id must be a number from 1 to 5')
        return None

    opt = {'k': params.k,
           'f': params.f,
           'sim_id': params.sim_id,
           'training_data': params.training_data,
           'data_to_classify': params.data_to_classify,
           'mode': params.unseen
           }
    return opt


if __name__ == '__main__':
    main()
