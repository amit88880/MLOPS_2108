# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt
import random
import numpy as np
# import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import preprocess_digits, data_viz, train_dev_test_split, h_param_tuning, get_all_h_params_comb, train_save_model
from joblib import dump,load

# 1. set the range of hyperparameters
gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


params = {}
params['gamma'] = gamma_list
params['C'] = c_list



# 2. for every combiation of hyper parameter values
h_param_comb = get_all_h_params_comb(params)

assert len(h_param_comb) == len(gamma_list)*len(c_list)

def h_param():
    return h_param_comb



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

# X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
#     data, label, train_frac, dev_frac, 1 - (train_frac + dev_frac), random_state = 5
# )

# X_train, y_train, X_test, y_test = train_test_split(
#     data, label, test_size=0.2, random_state = 5
# )


svmac = []
treeac = []

for i in range(5):

    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac, 1 - (train_frac + dev_frac)
    )

    model_path, clf = train_save_model(X_train, y_train, X_dev, y_dev, None, h_param_comb)
    
    treeclf = tree.DecisionTreeClassifier()
    treeclf =treeclf.fit(X_train, y_train)

    treepre =treeclf.predict(X_test)






    best_model = load(model_path)


    predicted= best_model.predict(X_test) 

    # PART: sanity check visulization of data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    svmac.append(accuracy_score(y_test, predicted))
    treeac.append(accuracy_score(y_test,treepre))


# PART: Compute evaluation Matrics 
# 4. report the best set accuracy with that best model.

print(f'SVM: {svmac}')
print(f'TREE: {treeac}')


svmac = np.array(svmac)
treeac = np.array(treeac)

svmmean = np.mean(svmac)
treemean = np.mean(treeac)

svm_variance = np.var(svmac)
tree_variance = np.var(treeac)

print(f"SVM Mean: {svmmean} \t Variance: {svm_variance}")
print(f'TREE Mean: {treemean} \t Variance: {tree_variance}')
print()
if svmmean > treemean:
    print("SVM is best")
    # print(
    #     f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # ) 

    # print(f"Best hyperparameters were: {best_model}")

else:
    print("Decision tree is best")
print()