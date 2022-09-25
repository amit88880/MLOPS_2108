# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt

# import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preprocess_digits(dataset):
    # PART: data pre-processing -- to normlize data, to remove noice, formate the data to be consumed by model
    samples = len(dataset.images)
    data = digits.images.reshape((samples, -1))
    label = dataset.target
    return data, label


# PART: sanity check visulization of data
def viz_data(dataset):
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
    best_param = None
    best_model = None
    accuracy = 0
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
        curr_predicted = clf.predict(X_test)


        # 2b. compute accuracy on validation set
        curr_accuracy = accuracy_score(y_test, curr_predicted)

        # 3. identify best set of hyper parameter for which validation set acuuracy is highest
        if accuracy < curr_accuracy:
            best_param = hyper_param
            accuracy = curr_accuracy
            best_model = clf
            print(f"{best_param} \tAccuracy: {accuracy}")

    return best_model, accuracy, best_param

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
data_viz(digits)
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
best_model, best_metric, best_param = h_param_tuning(hyp_para_combo, clf, X_train, y_train, X_dev, y_dev, metric)
predicted = best_model.predict(X_test) 

# PART: sanity check visulization of data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
    
# PART: Compute evaluation Matrics.
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n") 

print(f"Best hyperparameters were: {best_param}")
