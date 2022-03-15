from hw3 import perceptron
import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data

feature_set1=[('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)] 

feature_set2=[('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]
            

Ts = [1,10,50]
learners = [hw3.perceptron,hw3.averaged_perceptron]

def evaluate_auto(learner,T,feature_set):
    auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, feature_set)
    print(f" T: {T} result: {hw3.xval_learning_alg(lambda data,labels: learner(data,labels,{'T':T}),auto_data,auto_labels,10)}")

feature_set = feature_set1
#following 4.1 no longer needed
"""
print ("feature set 1: \n")
for T in Ts:
    for learner in learners:
        evaluate_auto(learner,T,feature_set)

feature_set = feature_set2
print ("feature set 2: \n")
for T in Ts:
    for learner in learners:
        evaluate_auto(learner,T,feature_set)
"""
#following for 4.2 A)
#evaluate_auto(hw3.averaged_perceptron,1,feature_set2)

"""following for 4.2 B) find 2 features that can produce comparable performance (0.9005)
"""
"""
feature_set_min=[('cylinders', hw3.one_hot),
            ('weight', hw3.standard)]
#cylinders and weight gives 0.8953205128205128
print("\nusing feature set min\n")
evaluate_auto(hw3.averaged_perceptron,1,feature_set_min)
"""

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data

def evaluate_review(learner,data,labels,T,dictionary):
    print(f" T: {T} result: {hw3.xval_learning_alg_review(lambda d,l: learner(d,l,{'T':T}),data,labels,10,dictionary,review_texts)}")

#following for 5.1
"""
Ts = [1,10,50]
learners = [hw3.perceptron,hw3.averaged_perceptron]

for T in Ts:
    for learner in learners:
        evaluate_review(learner,review_bow_data,review_labels,T)
"""

#5.2 find the 10 "best" and "worst" words
score_sum = hw3.eval_classifier_review(lambda d,l: hw3.averaged_perceptron(d,l,{'T':10}),review_bow_data,review_labels, review_bow_data,review_labels,dictionary,review_texts)
print(score_sum)

#5.2 extra find the 10 best and worst reviews
"""

def positive(x, th, th0):
    return np.sign(th.T@x + th0)

def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)

I need to have (th.T@x + th0) - have as list with index

"""



#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples,m,n = np.shape(x)
    flattened = np.reshape((x),(-1,m*n)) #(nsamples,m*n)
    out = np.reshape((flattened.T),(m*n,-1))
    return out

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    #TODO: modify for 3d input
    out = []
    for sample in x:
        #out.append(np.reshape(np.average(sample,axis=1),(-1,1)))
        out.append(np.average(sample,axis=1))
    return np.array(out).T

print("row average:\n")
print(row_average_features(np.array([[[1,2,3],[3,9,2]],[[6,7,8],[3,9,2]]])).tolist())

def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    out = []
    for sample in x:
        #TODO: modify for 3d input
        #out.append(np.reshape(np.average(sample,axis=0),(-1,1)))
        out.append(np.average(sample,axis=0))
    return np.array(out).T


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    n_samples,m,n = np.shape(x)
    #m,n = np.shape(x)
    out=[]
    #loops ARE allowed
    for sample in x:
        #print(sample)
        top_half = sample[0:int(np.floor(m/2))]
        btm_half = sample[int(np.floor(m/2)):]
        out.append([np.average(top_half),np.average(btm_half)])

    return np.array(out).T


#top_bottom_features(np.array([[[1,2,3],[4,5,6],[7,8,9]],[[4,5,6],[1,2,3],[1,2,3]]]))

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

def MNIST_task_accuracy(feature,d_0,d_1):

    d0 = mnist_data_all[d_0]["images"]
    d1 = mnist_data_all[d_1]["images"]
    y0 = np.repeat(-1, len(d0)).reshape(1,-1)
    y1 = np.repeat(1, len(d1)).reshape(1,-1)

    # data goes into the feature computation functions
    data = np.vstack((d0, d1))
    # labels can directly go into the perceptron algorithm
    labels = np.vstack((y0.T, y1.T)).T

    return hw3.get_classification_accuracy(feature(data), labels)

#6.2A

feature = raw_mnist_features

d_0 = 0
d_1 = 1
print(f"task {d_0} vs {d_1}: Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
d_0 = 2
d_1 = 4
print(f"task {d_0} vs {d_1}: Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
d_0 = 6 
d_1 = 8
print(f"task {d_0} vs {d_1}: Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
d_0 = 9 
d_1 = 0
print(f"task {d_0} vs {d_1}: Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")

feature = raw_mnist_features
d_0 = 0
d_1 = 1
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = row_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = col_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = top_bottom_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")

feature = raw_mnist_features
d_0 = 2
d_1 = 4
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = row_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = col_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = top_bottom_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")

feature = raw_mnist_features
d_0 = 6
d_1 = 8
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = row_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = col_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = top_bottom_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")

feature = raw_mnist_features
d_0 = 9
d_1 = 0
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = row_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = col_average_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")
feature = top_bottom_features
print(f"task {d_0} vs {d_1} Feature {feature.__name__} Accuracy = {MNIST_task_accuracy(feature,d_0,d_1)} /n")