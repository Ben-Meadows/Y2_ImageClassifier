#
# In this file please complete the following tasks:
#
# Task 4 [3] The curse of k
# Independent inquiry time! Picking the right number of neighbours k in the kNN approach is tricky.
# Find a way you could approach this more rigorously.
# In comments, state the approach you could use, and provide a reference to it.
# The reference needs to be to a handbook or peer-reviewed publication; a link to an online tutorial
# will not be accepted.
'''
I found many ways to find the optimal value of k such as cross-validation but the only one I could find and understand in a
peer-reviewed paper was the use of kTree. KTree’s are used via adding a training stage into the KNN process which outputs a
training model. kTree method first learns optimal values for all training samples by a new sparse reconstruction model, and
then constructs a decision tree using training samples and the learned optimal k values of each training sample. Once it finds
the optimal k values, it proceeds with traditional Knn classification. This Ktree can produce a time complexity of O(log(d) + n)
(where d is the dimensions of the features).

The two research papers I fond for this kTree are:

(‘APPROXIMATE K-NEAREST NEIGHBOUR BASED SPATIAL CLUSTERING USING K-D TREE; Dr Mohammed Otair’)
(‘Efficient kNN Classification With Different Numbers of Nearest Neighbours; Shichao Zhang, Senior Member, IEEE, Xuelong Li, Fellow, IEEE, Ming Zong, Xiaofeng Zhu, and Ruili Wang’)

'''


# Task 6 [3] I can do better!
# Independent inquiry time! There are much better approaches out there for image classification.
# Your task is to find one, and using the comment section of your project, do the following:
# •	State the name of the approach, and a link to a resource in the Cardiff University library that describes it
# •	Briefly explain how the approach you found is better than kNN in image classification (2-3 sentences is enough).
'''
There are many ways to tackle the problem of image classification, and some are better than others one type of image classification 
better than the KNN algorithm is the use of CNN’s, convolutional neural networks. These are a type of feed-forward artificial 
neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex, 
they work by extracting the features from the input data and finding similarities. But the big reason on why CNNs are better is 
because they require little dependence on pre-processing, decreasing the needs of human effort developing its functionalities, 
it also has the highest accuracy among all algorithms that predict images which is the main goal of all image classifiers.

The library resource I found:

(‘Developing an Image Classifier Using TensorFlow: Convolutional Neural Networks’, Kulshrestha, Saurabh,2019)

'''