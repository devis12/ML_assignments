# pylint: disable=E1136 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import learning_curve

def load_data(n_feat, filename_train_data, filename_train_targets, filename_test_data, filename_test_targets):

    vectorizer = HashingVectorizer(n_features=n_feat)

    #loading training data
    raw_train_data = (pd.read_csv(filename_train_data, header=None)).to_numpy()
    raw_train_targets = (pd.read_csv(filename_train_targets, header=None)).to_numpy()

    if(raw_train_data.shape[0] != raw_train_targets.shape[0]):
        print("ERROR!\nTraining samples not valid!")
        exit

    #process training data
    # learn new vocabulary from the training data & fit all the tweets into an occurencies matrix
    train_data = (vectorizer.fit_transform(raw_train_data[:,0])).toarray() 
    train_data = std_matrix_by_feature(train_data)


    #just transform DT to 1 and HC to 0
    train_targets = process_label(raw_train_targets)

    #loading testing data
    raw_test_data = np.array((pd.read_csv(filename_test_data, header=None)).values)
    raw_test_targets = np.array((pd.read_csv(filename_test_targets, header=None)).values)

    if(raw_test_data.shape[0] != raw_test_targets.shape[0]):
        print("ERROR!\nTesting samples not valid!")
        exit

    #process testing data (same processing as for the training data... just applied to the testing data)
    test_data = (vectorizer.transform(raw_test_data[:,0])).toarray()
    test_data = std_matrix_by_feature(test_data)
    test_targets = process_label(raw_test_targets)

    return train_data, train_targets, test_data, test_targets

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

def std_matrix_by_feature(data):
    for i in range(0,data.shape[1]):
        data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()
    return data


n_features_possibilities = [2**6, 2**7, 2**8, 2**9]

kf = KFold(n_splits=3, shuffle=True, random_state=42)

#best_selected_params
best_params = {
    'C': None,
    'gamma': None,
    'n_features': None
}

best_accuracy = 0

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Do model selection over all the possible values of gamma 
for n_feat in n_features_possibilities:
    
    print("\nK-FOLD\nn_features for HashingVectorizer = ", n_feat)
    train_data, train_targets, test_data, test_targets = load_data(n_feat,"tweets-train-data.csv","tweets-train-targets.csv","tweets-test-data.csv","tweets-test-targets.csv")
    #print(train_data.shape)

    possible_parameters = {
        'C': [1e0, 1e1, 1e2, 1e3],
        'gamma': [1e-1, 1e-2, 1e-3]
    }

    svc = SVC(kernel='rbf')
    clf = GridSearchCV(svc, possible_parameters, n_jobs=4, cv=3) # n_jobs=4 means we parallelize the search over 4 threads
    clf.fit(train_data, train_targets)
    
    best_C = clf.best_params_['C']
    best_gamma = clf.best_params_['gamma']
    
    # Train a classifier with current gamma
    clf = SVC(C=best_C, kernel='rbf', gamma=best_gamma)

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

    print(("Results for n_features = {0} K-FOLD cross validation (using C = {1}, gamma = {2})".format(n_feat, best_C, best_gamma)))
    print("Accuracy = ", accuracy_score)
    print("Precision = ", precision_score)
    print("Recall = ", recall_score)
    print("F1 = ", f1_score)


    if(accuracy_score > best_accuracy):
        best_accuracy = accuracy_score
        best_params['C'] = best_C
        best_params['gamma'] = best_gamma
        best_params['n_features'] = n_feat    
    

print(("\n\nAt the end of k-fold training, we select as n_features = {0} (C={1}, gamma={2})").format(best_params['n_features'], best_params['C'], best_params['gamma']))
train_data, train_targets, test_data, test_targets = load_data(best_params['n_features'],"tweets-train-data.csv","tweets-train-targets.csv","tweets-test-data.csv","tweets-test-targets.csv")
    

# Specify the parameters in the constructor.
# C is the parameter of the primal problem of the SVM;
# The rbf kernel is the Gaussian kernel;
# The rbf kernel takes one parameter: gamma (gaussian width)
clf = SVC(C=best_params['C'], kernel='rbf', gamma=best_params['gamma'])

# Training
clf.fit(train_data, train_targets)

# Prediction
predicted_targets = clf.predict(test_data)

#printing out final prediction results
report = metrics.classification_report(test_targets, predicted_targets)
print("REPORT:\n",report)
accuracy = metrics.accuracy_score(test_targets, predicted_targets)
print("Accuracy = ",accuracy)

#printing out plots with learning curve
plt.figure()
plt.title("Learning curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

# The function automatuically executes a Kfold cross validation for each dataset size
train_sizes, train_scores, val_scores = learning_curve(clf, train_data, train_targets, scoring='accuracy', cv=3)

# Get the mean and std of train and validation scores over the cv folds along the varying dataset sizes
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot the mean  for the training scores
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

# Plot the  std for the training scores
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")

# Plot the mean  for the validation scores
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

# Plot the std for the validation scores
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.ylim(0.05,1.3)             # set bottom and top limits for y axis
plt.legend()
plt.show()

#printing out predicted labels in the output file
df = pd.DataFrame(unprocess_label(predicted_targets))
df.to_csv("test-pred.txt", index=False, header=False)