# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

#PART-1: Library Dependencies

# Standard scientific Python imports
from ctypes import sizeof
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from skimage.transform import rescale
from sklearn.model_selection import train_test_split


# 1.Set ranges for hyperparameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001 ]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


h_param_comb = [{'gamma': g, 'C': c}for g in gamma_list for c in c_list]
assert len(h_param_comb) == len(gamma_list)*len(c_list)


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


#PART-2: LOAD DATASETS s
digits = datasets.load_digits()


#PART-2.1: Sanity Check Visualization of the data

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


n_samples = len(digits.images)



img_res_ls = [1, 0.5, 1.5, 2]

#best_acc, best_acc_train, best_acc_test = -1.0, -1.0, -1.0
#best_model = None
#best_h_params = None

print("Image res: " + str(digits.images[0].shape))

for i in img_res_ls:

    best_acc, best_acc_train, best_acc_test = -1.0, -1.0, -1.0
    best_model = None
    best_h_params = None
    #PART-5: Define train/test/dev splits of experiment protocol
    #train to train model
    #dev to set hyperparameters of the model
    #test to evaluate performance of the model (test on unseen data, to avoid overestimation of performance of the model)
    image_rescaled = rescale(image, i, anti_aliasing=False)
    print("Resized image res= " + str(image_rescaled.shape))

    data = digits.images.reshape((n_samples, -1))
    #model cannot take a 2d image as input so we convert every images into single array

    dev_test_frac = 1 - train_frac
    #Split into 80:10:10 :: train:dev:test
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size= dev_test_frac, shuffle=True
    )
    #Resplit
    #Split dev_test variables
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/(dev_test_frac), shuffle=True
    )
    
    #model hyperparameters
    #GAMMA = 0.001
    #C = 0.5
    ls = []

    for cur_h_params in h_param_comb:

        
        #PART-4: Define the model

        # Create a classifier: a support vector classifier
        clf = svm.SVC() #support vector machine

        #PART-4.1: Setting up the hyperparameters
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        #PART-6: Train the model
        # Learn the digits on the train subset

        # 2.Train the model for every combination of hyperparameters
        clf.fit(X_train, y_train)

        #print(cur_h_params)

        #PART-7: Get dev set predictions
        predicted_dev = clf.predict(X_dev)
        predicted_train = clf.predict(X_train)
        predicted_test = clf.predict(X_test)

        # 3.Compute the accuracy on the validation set
        cur_acc = metrics.accuracy_score(y_pred = predicted_dev, y_true=y_dev)
        cur_acc_train = metrics.accuracy_score(y_pred = predicted_train, y_true=y_train)
        cur_acc_test = metrics.accuracy_score(y_pred = predicted_test, y_true=y_test)

        
        row = [cur_h_params['gamma'], cur_h_params['C'], cur_acc_train, cur_acc, cur_acc_test]
        ls.append(row)
	
        # 4.Identify the best combination of hyperparameters for which validation set accuracy is highest
       	if cur_acc > best_acc:
            best_acc = cur_acc
            best_model = clf
            best_h_params = cur_h_params
            #print("Current best accuracy with : " + str(best_h_params))
            #print("New best validation accuracy :" + str(best_acc))
        if cur_acc_train > best_acc_train:
            best_acc_train = cur_acc_train
        
        if cur_acc_test > best_acc_test:
            best_acc_test = cur_acc_test


    
    print("Gamma\tC\t\ttrain acc\tdev acc\t\ttest acc")
    for i in ls:
        print(i)
    

    print("Best hyperparameters: " + str(best_h_params))
    print("Best train acc: " + str(best_acc_train))
    print("Best dev acc: " + str(best_acc))
    print("Best test acc: " + str(best_acc_test))
    print("\n")

#PART-7.1: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(X_test)


#PART-8: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


#print(cur_h_params)

# 5.Report the test set accuracy with the chosen combo of parameters

#PART-9: Compte Evaluation matrices
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.
'''
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
'''
print("Best hyperparameters were: " + str(cur_h_params))
