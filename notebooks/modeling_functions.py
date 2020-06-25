import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def encode_features(dataframe, columns):
    """Integer encoding of categorical features (columns)"""
    df = dataframe.copy()
    # for each categorical column
    for col in columns:
        le = LabelEncoder()
        integer_encoded = le.fit_transform(df[col])
        df[col] = integer_encoded

    return df, le

def choose_k(train_labels):
    """Determine number of folds
    """
    DEFAULT_K = 10
    
    class_counter = Counter(train_labels)
    num_least_common_class = min(class_counter.values())

    return min(num_least_common_class, DEFAULT_K)

def select_model_parameters(train_vectors, train_labels, trainclf, parameters, scorer, 
                          use_sample_weight = False):
    """Choose the best combination of parameters for a given model
    """
    
    k = choose_k(train_labels) # get number of folds

    stratifiedKFold = StratifiedKFold(n_splits = k)
    if use_sample_weight:
        n_samples = len(train_labels)
        n_classes = len(set(train_labels))
        classCounter = Counter(train_labels)
        sampleWeights = [n_samples / (n_classes * classCounter[label]) for label in train_labels]        
        gridSearch = GridSearchCV(trainclf, parameters, cv = stratifiedKFold, scoring = scorer, 
                                      fit_params = {'sample_weight' : sampleWeights})
    else:
        gridSearch = GridSearchCV(trainclf, parameters, cv = stratifiedKFold, scoring = scorer)
    
    gridSearch.fit(train_vectors, train_labels)
    print("Best parameters set found: {}".format(gridSearch.best_params_))
    
    return gridSearch.best_estimator_

def train_rdf(dataframe, plot=True):
    """Breakdown the dataframe into X and y arrays. Later, split them in CV and test set. Train 
    the model with CV and then find the optimal tree depth and report the accuracy.
    Assumes the last column is the label.
    """
    
    # create design matrix X and vector y
    X = np.array(dataframe.iloc[:, 0:dataframe.shape[1] - 1]) # minus 1 for the target label
    y = np.array(dataframe.iloc[:, -1]) # last column is the target label
    
    print("Features: {}".format(dataframe.columns.values[:-1]))  # minus 1 for the comfort label

    parameters = {'n_estimators' : [10, 100, 1000],
                  'criterion' : ['entropy', 'gini'],
                  'min_samples_split' : [2, 10, 20, 30], 
                  'class_weight' : ['balanced', 'balanced_subsample']}
    scorer = 'f1_micro'
    clf = RandomForestClassifier(n_estimators = 10, min_samples_split = 2, class_weight = 'balanced', 
                                 random_state = 13)
    
    # split into CV and test
    test_size_percentage = 0.2

    # X_train = cv set (train_vectors)
    # X_test = test set (test_vectors)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_percentage, 
                                                        random_state = 13, stratify = y)
    # find best parameters with gridsearchCV
    rf_classifier = select_model_parameters(X_train, y_train, clf, parameters, scorer)
    # find optimal depth and generate model
    optimal_depth = find_optimal_tree_depth(rf_classifier, X_train, y_train, plot)
    # generate the model with the selected paramters plus the optimal depth and do the model fitting
    rf_optimal = rf_classifier.set_params(max_depth = optimal_depth)
    print(rf_optimal)
    
    # fit and predict on the test set
    rf_optimal.fit(X_train, y_train)
    y_pred = rf_optimal.predict(X_test)

    # get metrics
    rf_acc = get_clf_metrics(y_test, y_pred)
    
    return rf_optimal

def find_optimal_tree_depth(clf, train_vectors, train_labels, plot=True):
    """Choose the optimal depth of a tree model 
    """
    
    DEFAULT_K = 10
    
    # generate a list of potential depths to calculate the optimal
    depths = list(range(1, 25))

    # empty list that will hold cv scores
    cv_scores = []

    print("Finding optimal tree depth")
    # find optimal tree depth    
    for d in depths:
        clf_depth = clf.set_params(max_depth = d) # use previous parameters while changing depth

        scores = cross_val_score(clf_depth, train_vectors, 
                                 train_labels, cv = choose_k(train_labels),
                                 scoring = 'accuracy') # accuracy here is f1 micro
        cv_scores.append(scores.mean())

    # changing to misclassification error and determining best depth
    MSE = [1 - x for x in cv_scores] # MSE = 1 - f1_micro
    optimal_depth = depths[MSE.index(min(MSE))]
    
    print("The optimal depth is: {}".format(optimal_depth))
    print("Expected accuracy (f1 micro) based on Cross-Validation: {}".format(cv_scores[depths.index(optimal_depth)]))
    
    if plot:
        # plot misclassification error vs depths
        fig = plt.figure(figsize=(12, 10))
        plt.plot(depths, MSE)
        plt.xlabel('Tree Depth', fontsize = 20)
        plt.ylabel('Misclassification Error', fontsize = 20)
        plt.legend(fontsize = 15)
        plt.show()

    return optimal_depth

def comfPMV(ta, tr, vel, rh, met, clo, wme):
    """From https://github.com/CenterForTheBuiltEnvironment/comfort_tool/blob/master/contrib/comfort_models.py

    Parameters
    ----------
    ta, air temperature (C)
    tr, mean radiant temperature (C)
    vel, relative air velocity (m/s)
    rh, relative humidity (%) Used only this way to input humidity level
    met, metabolic rate (met)
    clo, clothing (clo)
    wme, external work, normally around 0 (met)
    
    Returns
    ----------
    [pmv, ppd]
    """

    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

    icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    m = met * 58.15  # metabolic rate in W/M2
    w = wme * 58.15  # external work in W/M2
    mw = m - w  # internal heat production in the human body
    if (icl <= 0.078):
        fcl = 1 + (1.29 * icl)
    else:
        fcl = 1.05 + (0.645 * icl)

    # heat transf. coeff. by forced convection
    hcf = 12.1 * math.sqrt(vel)
    taa = ta + 273
    tra = tr + 273
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * math.pow(tra / 100, 4))
    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
        if (hcf > hcn):
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
        n += 1
        if (n > 150):
            print('Max iterations exceeded')
            return 1


    tcl = 100 * xn - 273

    # heat loss diff. through skin
    hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    # heat loss by sweating
    if mw > 58.15:
        hl2 = 0.42 * (mw - 58.15)
    else:
        hl2 = 0
    # latent respiration heat loss
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    # dry respiration heat loss
    hl4 = 0.0014 * m * (34 - ta)
    # heat loss by radiation
    hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100, 4))
    # heat loss by convection
    hl6 = fcl * hc * (tcl - ta)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0)
        - 0.2179 * pow(pmv, 2.0))

    r = []
    r.append(pmv)
    r.append(ppd)

    return r

def get_clf_metrics(test_labels, pred_labels):
    """Compute different validation metrics for a classification task.
    """
    
    # because it's multi-class, acc and f1_micro are calculated in the same way
    acc = accuracy_score(test_labels, pred_labels)
    f1_micro = f1_score(test_labels, pred_labels, average = 'micro')
    f1_macro = f1_score(test_labels, pred_labels, average = 'macro')
    print("Accuracy (f1 micro) on validation set: {}".format(acc))
    print("F1 micro on validation set: {}".format(f1_micro))
    print("F1 macro on validation set: {}".format(f1_macro))
    print("Confusion Matrix: ")
    print(confusion_matrix(test_labels, pred_labels))
    print("Classification Metrics: ")
    print(classification_report(test_labels, pred_labels))
    
    return f1_micro, f1_macro

def model_validate(df_train, df_test, clf_optimal):
    # transform unseen test set and whole train split
    X_train = np.array(df_train.iloc[:, 0:df_train.shape[1] - 1]) # minus 1 for the comfort label
    y_train = np.array(df_train.iloc[:, -1])

    X_test = np.array(df_test.iloc[:, 0:df_test.shape[1] - 1]) # minus 1 for the comfort label
    y_test = np.array(df_test.iloc[:, -1])
      
    # retrain in all train split with the tuned model
    clf_optimal.fit(X_train, y_train)

    #predict the response on test set
    y_pred = clf_optimal.predict(X_test)

    # get metrics
    f1_micro, f1_macro = get_clf_metrics(y_test, y_pred)
    
    return f1_micro, f1_macro, y_pred

def find_unitary_class(y):
    class_counter = Counter(y)
#     print(class_counter)
    num_least_common_class = min(class_counter.values())
    if num_least_common_class == 1:
#         label = class_counter.keys()[class_counter.values().index(num_least_common_class)]
        label = list(class_counter.keys())[list(class_counter.values()).index(num_least_common_class)]
    else:
        label = -10
    return label

def remove_unitary_row(dataframe, column_name, participant):
    """Checks for unitary sample in a dataframe, removes it, and returns a dataframe without
    that sample"""
    
    df = dataframe.copy()
    
    # checks for unitary label
    label_to_remove = find_unitary_class(dataframe[column_name])
    
    # value used when no unitary class was find
    if label_to_remove != -10:
        df = df[df[column_name] != label_to_remove]
        print("Participant " + str(participant) + " has label " + str(label_to_remove) + " as unitary class.\n")
    
    return df
        