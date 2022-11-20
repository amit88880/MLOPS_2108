import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from joblib import dump,load


def get_all_h_params_comb(params):
    hyp_para_comb = [{"gamma":g, "C":c} for g in params['gamma'] for c in params['C']]
    return hyp_para_comb

def preprocess_digits(ds):
    # PART: data pre-processing -- to normlize data, to remove noice,
    #                               formate the data tobe consumed by model
    n_samp = len(ds.images)
    data = ds.images.reshape((n_samp, -1))
    label = ds.target
    return data, label


def data_viz(ds):
    # PART: sanity check visulization of data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, ds.images, ds.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

def train_dev_test_split(data, label, train_frac, dev_frac, test_frac,randomstate):
    dev_test_frac = 1- train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True, random_state = randomstate
    )


    fraction_want = dev_frac/(dev_frac+test_frac)
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=fraction_want, shuffle=True, random_state = randomstate
    )

   
    return X_train, y_train, X_dev, y_dev, X_dev, y_dev



def h_param_tuning(hyp_para_combo, clf, X_train, y_train, X_dev, y_dev, metric):
    best_hyp_param = None
    best_model = None
    accuracy = 0
    for curr_param in hyp_para_combo:
        # PART: setting up hyper parameter
        hyper_param = curr_param
        clf.set_params(**hyper_param)

        # PART: train model
        # 2a. train the model
        # Learn the ds on the train subset
        clf.fit(X_train, y_train)


        # PART: get test set pridection
        # Predict the value of the digit on the test subset
        current_pred = clf.predict(X_dev)


        # 2b. compute accuracy on validation set
        curr_accuracy = metric(y_dev, current_pred)

        # 3. identify best set of hyper parameter for which validation set acuuracy is highest
        if accuracy < curr_accuracy:
            best_hyp_param = hyper_param
            accuracy = curr_accuracy
            best_model = clf
            # print(f"{best_hyp_param} \tAccuracy: {accuracy}")

    return best_model, accuracy, best_hyp_param


def train_save_model(X_train, y_train, X_dev, y_dev, modelpath, h_param_comb):
    

    # PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()
    metric = metrics.accuracy_score
    best_model, best_metric, best_hyp_param = h_param_tuning(h_param_comb, clf, X_train, y_train, X_dev, y_dev, metric)
    # if predicted < current_pred:
    #     predicted = current_pred


    best_param_config = "_".join([h+"_"+str(best_hyp_param[h]) for h in best_hyp_param])

    if modelpath is None:
        modelpath = "svm_" + best_param_config + ".joblib"

    dump(best_model, "svm_" + best_param_config + ".joblib")


    return modelpath, clf

## random state is fixed .
##  for same test and train data.
def test_split_same():
    train_frac = 0.8
    test_frac = 0.1
    dev_frac = 0.1

    # actual dataset.
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 150
    )

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 150
    )

    # checking
    assert (X_train1 == X_train2).all()
    assert (X_test1 == X_test2).all()
    assert (y_train1 == y_train2).all()
    assert (y_test1 == y_test2).all()


    ## if random state is different.

def test_split_same1():
    train_frac = 0.7
    test_frac = 0.15
    dev_frac = 0.15

    # actual dataset.
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 150
    )

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 150
    )

    # checking
    assert (X_train1 == X_train2).all()
    assert (y_train1 == y_train2).all()
    assert (X_test1 == X_test2).all()
    assert (y_test1 == y_test2).all()

# perf_test = {}
# for k in range(S):
#     train, dev, test = create_split()
#     best_model = train_and_h_tune()