# pylint: disable=E1136 
import numpy as np
import pandas as pd

import sklearn
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold, cross_val_score

#convert "DT" to class 1 (true) and everything else to class 0
def process_label(output_data):
    for count in range(0, len(output_data)):
        if(output_data[count] == "DT"):
            output_data[count] = 1
        else:
            output_data[count] = 0
    return output_data[:,0].astype("int")

#convert class 1 to "DT" (true) and the others to "HC"
def unprocess_label(output_data):
    res = []
    for count in range(0, len(output_data)):
        if(output_data[count] == 1):
            res.append("DT")
        else:
            res.append("HC")
    return res


#object in sklearn useful to vectorize an array of strings into a matrix of occurencies for the words within the given strings
vectorizer = CountVectorizer()
#vectorizer = HashingVectorizer(n_features=2**8)

#loading training data
raw_train_data = (pd.read_csv("tweets-train-data.csv", header=None)).to_numpy()
raw_train_targets = (pd.read_csv("tweets-train-targets.csv", header=None)).to_numpy()
print("# of examples in raw training data = ", len(raw_train_data))
print("# of examples in raw training labels = ", len(raw_train_targets))

if(raw_train_data.shape[0] != raw_train_targets.shape[0]):
    print("ERROR!\nTraining samples not valid!")
    exit

#process training data
# learn new vocabulary from the training data & fit all the tweets into an occurencies matrix
train_data = (vectorizer.fit_transform(raw_train_data[:,0])).toarray() 
#just transform DT to 1 and HC to 0
train_targets = process_label(raw_train_targets)
print("# of examples in processed training data = ", len(train_data))
print("# of examples in processed training labels = ", len(train_targets))

#loading testing data
raw_test_data = np.array((pd.read_csv("tweets-test-data.csv", header=None)).values)
raw_test_targets = np.array((pd.read_csv("tweets-test-targets.csv", header=None)).values)
print("# of examples in raw testing data = ", len(raw_test_data))
print("# of examples in raw testing labels = ", len(raw_test_targets))

if(raw_test_data.shape[0] != raw_test_targets.shape[0]):
    print("ERROR!\nTesting samples not valid!")
    exit

#process testing data (same processing as for the training data... just applied to the testing data)
test_data = (vectorizer.transform(raw_test_data[:,0])).toarray() 
test_targets = process_label(raw_test_targets)
print("# of examples in processed testing data = ", len(test_data))
print("# of examples in processed testing labels = ", len(test_targets))

print(train_data.shape)
print(test_data.shape)


kf = KFold(n_splits=3, shuffle=True, random_state=42)

gamma_values = [0.1, 0.02, 0.001]

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Do model selection over all the possible values of gamma 
for gamma in gamma_values:
    
    # Train a classifier with current gamma
    clf = SVC(C=10, kernel='rbf', gamma=gamma)

    # Compute cross-validated accuracy scores
    acc_scores = cross_val_score(clf, train_data, train_targets, cv=kf.split(train_data), scoring='accuracy')
    # Compute the mean accuracy and keep track of it
    accuracy_score = acc_scores.mean()
    accuracy_scores.append(accuracy_score)

    # Compute cross-validated precision scores
    pr_scores = cross_val_score(clf, train_data, train_targets, cv=kf.split(train_data), scoring='precision')
    # Compute the mean precision and keep track of it
    precision_score = pr_scores.mean()
    precision_scores.append(precision_score)

    # Compute cross-validated recall scores
    rec_scores = cross_val_score(clf, train_data, train_targets, cv=kf.split(train_data), scoring='recall')
    # Compute the mean recall and keep track of it
    recall_score = rec_scores.mean()
    recall_scores.append(recall_score)

    # Compute cross-validated f1 scores
    f1_scores_temp = cross_val_score(clf, train_data, train_targets, cv=kf.split(train_data), scoring='f1')
    # Compute the mean recall and keep track of it
    f1_score = f1_scores_temp.mean()
    f1_scores.append(f1_score)
    
# Get the gamma with highest mean accuracy
best_index = np.array(accuracy_scores).argmax()
best_gamma = gamma_values[best_index]    

# Specify the parameters in the constructor.
# C is the parameter of the primal problem of the SVM;
# The rbf kernel is the Gaussian kernel;
# The rbf kernel takes one parameter: gamma (gaussian width)
clf = SVC(C=10, kernel='rbf', gamma=best_gamma)

# Training
clf.fit(train_data, train_targets)

# Prediction
predicted_targets = clf.predict(test_data)

report = metrics.classification_report(test_targets, predicted_targets)
print("REPORT:\n",report)
accuracy = metrics.accuracy_score(test_targets, predicted_targets)
print("Accuracy = ",accuracy)

df = pd.DataFrame(unprocess_label(predicted_targets))
df.to_csv("test-pred.txt", index=False, header=False)