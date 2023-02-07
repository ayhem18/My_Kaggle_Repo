# import packages
import pickle

import numpy as np
# import models
from sklearn.linear_model import LogisticRegression
# import metrics and scoring modules
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, balanced_accuracy_score
# import tuning modules
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

score_function = {"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score,
                  "f1": f1_score, "balanced_accuracy": balanced_accuracy_score}

# constant used for cross validation
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)


# get_scorer_names()


def tune_model(model, params_grid, X_train, y_train, cv=None, scoring=None):
    # scoring should be determined depending on the nature of the classification problem
    if scoring is None:
        # if the classification problem is binary
        if len(np.unique(y_train)) == 2:
            scoring = 'f1'
        else:
            scoring = 'balanced_accuracy'

    if cv is None:
        cv = CV

    searcher = GridSearchCV(model, param_grid=params_grid, cv=cv, scoring=scoring, n_jobs=-1)
    searcher.fit(X_train, y_train)
    return searcher.best_estimator_


def evaluate_tuned_model(tuned_model, X_train, X_test, y_train, y_test, train=True, metrics=None):
    if metrics is None:
        metrics = ['accuracy']
    if isinstance(metrics, str):
        metrics = [metrics]

    # train the model
    if train:
        tuned_model.fit(X_train, y_train)

    # predict on the test dataset
    y_pred = tuned_model.predict(X_test)
    # evaluate the model
    scores = dict(list(zip(metrics, [score_function[m](y_test, y_pred) for m in metrics])))
    return tuned_model, scores


def save_model(tuned_model, path):
    with open(path, 'wb') as f:
        pickle.dump(tuned_model, f)


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def try_model(model, X, y, params_grid, save=True, save_path=None, test_size=0.2, tune_metric=None,
              test_metrics=None, cv=None):
    # the dataset passed is assumed to be ready to be processed
    # all its features are numerical and all its missing values are imputed/discarded


    if save and save_path is None:
        raise ValueError("Please pass a path to save the model or set the 'save' parameter to False")

    # split the dataset into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=11, stratify=y)

    # tune the model
    tuned_model = tune_model(model, params_grid, X_train, y_train, cv=cv, scoring=tune_metric)

    # evaluate teh tuned model
    model, results = evaluate_tuned_model(tuned_model, X_train, X_test, y_train, y_test, metrics=test_metrics)
    # save the model to the passed path
    if save:
        save_model(tuned_model, save_path)

    return model, results


lr_basic = LogisticRegression(max_iter=5000)

LR_grid = {"C": [0.1]}


def try_linear_regression(X, y, lr_model=lr_basic, params_grid=None, save=True, save_path=None,
                          test_size=0.2, tune_metric=None, test_metrics=None, cv=None):
    if params_grid is None:
        params_grid = LR_grid
    if test_metrics is None:
        test_metrics = ['accuracy']
    return try_model(lr_model, X, y, params_grid, save=save, save_path=save_path,
                     test_size=test_size, tune_metric=tune_metric, test_metrics=test_metrics, cv=cv)


if __name__ == "__main__":
    X, Y = make_classification(n_samples=4000, n_features=20, n_classes=3, random_state=18, n_informative=8)

    lr, results = try_linear_regression(X, Y, save=False)

    print(results)
