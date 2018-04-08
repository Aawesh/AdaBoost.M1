import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""Returs error given the prediction and the input"""
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

"""A function that takes classifier as input and performs classification"""
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)
    
"""Adaboot algorithm implementation"""
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    alpha_m = 0

    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test),clf,alpha_m

"""Function to plot error vs number of iterations"""
def plot_error_rate(er_train, er_test):
    plt.plot(er_train, label="Training")
    plt.plot(er_test, label="Test")
    plt.xlabel('Number of iterations')
    plt.legend()
    plt.title('Error rate vs number of iterations')
    plt.grid()
    plt.show()

"""Remove nulls, drops duplicates and convert to numbers """
def formatData(data):
    data = data.fillna(0.0)
    data.drop_duplicates()

    data.loc[data["Sex"] == "male",'Sex'] = 0
    data.loc[data["Sex"] == "female",'Sex'] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S",'Embarked'] = 0
    data.loc[data["Embarked"] == "C",'Embarked'] = 1
    data.loc[data["Embarked"] == "Q",'Embarked'] = 2

    return data

def adaboost_predict(inputs,hypotheses, hypotheses_weight):
    y = 0
    for (h, alpha) in zip(hypotheses, hypotheses_weight):
        y = y + alpha * h.predict(inputs)
    return np.sign(y)

"""Driver program"""
if __name__ == '__main__':

    data = pd.read_csv("titanic_data.csv")
    data = formatData(data)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","Survived"]
    data = data[features]

    # Split into training and test set
    train, test = train_test_split(data, test_size = 0.2)
    X_train, Y_train = train.ix[:,:-1], train.ix[:,-1]
    X_test, Y_test = test.ix[:,:-1], test.ix[:,-1]

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)

    # Fit Adaboost classifier using a decision tree stump
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    h,h_w = [],[]
    x_range = range(10, 1000, 10)
    for i in x_range:
        op = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        er_train.append(op[0])
        er_test.append(op[1])
        h.append(op[2])
        h_w.append(op[3])

    test = pd.read_csv("test_data.csv")
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
    test = test[features]
    test = test.ix[:, :-1]

    #Should give -1 if correct and in fact it does
    print adaboost_predict(test, h, h_w)

    # Compare error rate vs number of iterations
    plot_error_rate(er_train, er_test)


