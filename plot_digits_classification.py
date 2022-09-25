# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preprocess_digits(dataset):
    # PART: data pre-processing -- to normlize data, to remove noice,
    #                               formate the data tobe consumed by model
    n_samples = len(dataset.images)
    data = digits.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label


def viz_data(dataset):
    # PART: sanity check visulization of data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


def train_test(data, label, train_frac, dev_frac, test_frac):
    dev_test_frac = 1- train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )


    fraction_want = dev_frac/(dev_frac+test_frac)
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=fraction_want, shuffle=True
    )

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def h_param_tuning(hyp_para_combo, clf, X_train, y_train, X_dev, y_dev, metric):
    train_acc = []
    test_acc = []
    dev_acc = []
    best_hyp_param = None
    best_model = None
    best_train_acc, best_dev_acc, best_test_acc  = 0, 0, 0
    for curr_param in hyp_para_combo:
        # PART: setting up hyper parameter
        hyper_param = curr_param
        clf.set_params(**hyper_param)

        # PART: train model
        # 2a. train the model
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)


        # PART: get test set pridection
        # Predict the value of the digit on the test subset
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        dev_predict = clf.predict(X_dev)
        # curr_predicted = clf.predict(X_test)


        # 2b. compute accuracy on validation set
        curr_test_accuracy = accuracy_score(y_test, test_predict)
        curr_train_accuracy = accuracy_score(y_train, train_predict)
        curr_dev_accuracy = accuracy_score(y_dev, dev_predict)

        # 3. identify best set of hyper parameter for which validation set acuuracy is highest
        if best_train_acc < curr_train_accuracy:
            best_hyp_param = hyper_param
            best_train_acc = curr_train_accuracy
            best_model = clf
            print(f"{best_hyp_param} \ttrain_Accuracy: {best_train_acc}")

        if best_test_acc < curr_test_accuracy:
            best_test_acc = curr_test_accuracy
            print(f"{best_hyp_param} \ttest_Accuracy: {best_test_acc}")

        if best_dev_acc < curr_dev_accuracy:
            best_dev_acc = curr_dev_accuracy
            print(f"{best_hyp_param} \tdev_Accuracy: {best_dev_acc}")

        train_acc.append(best_train_acc)
        test_acc.append(best_test_acc)
        dev_acc.append(best_dev_acc)

    return best_model, best_train_acc, best_hyp_param, train_acc, test_acc, dev_acc

# 1. set the range of hyperparameters
gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


# 2. for every combiation of hyper parameter values
hyp_para_combo = [{"gamma":g, "C":c} for g in gamma_list for c in c_list]

assert len(hyp_para_combo) == len(gamma_list)*len(c_list)



train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1


# PART: -load dataset --data from files, csv, tsv, json, pickle
digits = datasets.load_digits()
viz_data(digits)
data, label = preprocess_digits(digits)

# housekeeping
del digits



# PART: define train/dev/test splite of experiment protocol
# trin to train model
# dev to set hyperperameter of the model
# test to evaluate the performane of the model

# 80:10:10 train:dev:test


# if testing on the same as traning set: the performance matrics may overestimate 
# the goodness of the model. 
# We want to test on "unseen" sample. 

X_train, y_train, X_dev, y_dev, X_test, y_test = train_test(
    data, label, train_frac, dev_frac, test_frac
)
  

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
metric = metrics.accuracy_score
best_model, best_metric, best_hyp_param, train_acc, test_acc, dev_acc = h_param_tuning(hyp_para_combo, clf, X_train, y_train, X_dev, y_dev, metric)
# if predicted < curr_predicted:
#     predicted = curr_predicted

predicted_test = best_model.predict(X_test) 

# PART: sanity check visulization of data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
# PART: Compute evaluation Matrics 
# 4. report the best set accuracy with that best model.
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted_test)}\n"
) 

print(f"Best hyperparameters were: {best_hyp_param}")


dff = {'train_acc': train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}

table = pd.DataFrame.from_dict(dff)

print(table)