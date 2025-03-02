import marimo

__generated_with = "0.11.2"
app = marimo.App(
    width="medium",
    app_title="ML_pipeline",
    auto_download=["html"],
)


@app.cell
def _():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Importing libraries and data""")
    return


@app.cell
def _():
    import marimo as mo

    # Data Manipulation
    import numpy as np
    import scipy as sp
    import pandas as pd

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Machine Learning
    import sklearn
    import tensorflow as tf
    import tensorflow.keras as keras
    return keras, mo, np, pd, plt, sklearn, sns, sp, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Dataset""")
    return


@app.cell
def _(mo, os, pd):
    _dataset_path = os.path.join(mo.notebook_dir(), 'public', 'diabetes.csv')
    data = pd.read_csv(_dataset_path)
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Target""")
    return


@app.cell
def _(data):
    y = data['Outcome']
    y
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Features""")
    return


@app.cell
def _(data):
    X = data.drop(['Outcome'], axis=1)
    X
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Preprocessing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **1. Categorical feature** - Data does not have any string or categorical attributes. No such processing required.

        **2. Check imbalance ratio (IR)** : If IR is high (>3), other processing (e.g. data resampling, cost-sensitive learning) is required to address the imbalance. Otherwise, classification performance can become biased.
        """
    )
    return


@app.cell
def _(y):
    y.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Check Missing Entries""")
    return


@app.cell
def _(X):
    X.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Types""")
    return


@app.cell
def _(data):
    data.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Check Feature Importance""")
    return


@app.cell
def _(data):
    co = data.corr()
    co
    return (co,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Heatmap""")
    return


@app.cell
def _(co, sns):
    sns.heatmap(co, annot=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Do not have any duplicate or redundant features.""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Training Classifiers""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Validation scheme""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        1. Train Test Split
        2. K fold Cross Validation
        3. Statified K fold CV 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        *   Large dataset (e.g. Image data) - train-test split
        *   Small dataset (structured data) - cross validation (CV)
        *   Very large dataset (structured data) - repeat 3 times

        => depending on dataset size and other issues like imbalance ratio, you may use different CV schemes (will be discussed in detail in class).

        * Small dataset -> cross-validation (cv)
        * small imbalance -> stratified cv
        * good number of minority class instances -> 10 fold
        * small number of minority class instances -> 5 fold

        * best practice -> repeated stratified CV (5 repeats, 10 fold)
        """
    )
    return


@app.cell
def _():
    # Import Validation modules
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold, cross_validate, train_test_split


@app.cell
def _(StratifiedKFold):
    sk = StratifiedKFold(shuffle = True, random_state = 100, n_splits= 10)
    return (sk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Training""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Put all steps inside the pipeline - it'll divide the data internally into training and testing folds; perform operations (scaling, sampling, etc.) only on the training set and measure performance on the test set.""")
    return


@app.cell
def _():
    # Import Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    from xgboost import XGBClassifier
    return (
        DecisionTreeClassifier,
        GaussianNB,
        KNeighborsClassifier,
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        XGBClassifier,
    )


@app.cell
def _(
    DecisionTreeClassifier,
    GaussianNB,
    KNeighborsClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    XGBClassifier,
):
    knn= KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    lr = LogisticRegression()
    dt= DecisionTreeClassifier(random_state=10)
    svc = SVC(random_state=10)
    nb = GaussianNB()
    rf = RandomForestClassifier(random_state=10, n_jobs= -1)
    xgb = XGBClassifier(random_state=10)
    return dt, knn, lr, nb, rf, svc, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Pipeline""")
    return


@app.cell
def _():
    # Import pipeline
    from sklearn.pipeline import Pipeline

    # Import Scaler
    from sklearn.preprocessing import StandardScaler
    return Pipeline, StandardScaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Use cross_validate for validation when u want multiple performance measures.

        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
        """
    )
    return


@app.cell
def _(X, cross_validate, pd, sk, y):
    def get_model_result(pipe, name, X=X, y=y, sk=sk, scoring=None):
        _result = cross_validate(pipe, X, y, cv = sk, n_jobs = -1 , scoring=scoring)
        _df = pd.DataFrame(_result).mean()
        return pd.DataFrame(_df, columns=[name])
    return (get_model_result,)


@app.cell
def _(
    Pipeline,
    StandardScaler,
    dt,
    get_model_result,
    knn,
    lr,
    nb,
    pd,
    rf,
    svc,
    xgb,
):
    pipe_knn = Pipeline([ ('scaler', StandardScaler()), ('knn', knn)])
    _result_knn = get_model_result(pipe_knn, 'knn')

    pipe_lr = Pipeline([ ('scaler', StandardScaler()), ('lr', lr)])
    _result_lr = get_model_result(pipe_lr, 'lr')

    pipe_dt = Pipeline([ ('scaler', StandardScaler()), ('dt', dt)])
    _result_dt = get_model_result(pipe_dt, 'dt')

    pipe_svc = Pipeline([ ('scaler', StandardScaler()), ('svc', svc)])
    _result_svc = get_model_result(pipe_svc, 'svc')

    pipe_nb = Pipeline([ ('scaler', StandardScaler()), ('nb', nb)])
    _result_nb = get_model_result(pipe_nb, 'nb')

    pipe_rf = Pipeline([ ('scaler', StandardScaler()), ('rf', rf)])
    _result_rf = get_model_result(pipe_rf, 'rf')

    pipe_xgb = Pipeline([ ('scaler', StandardScaler()), ('xgb', xgb)])
    _result_xgb = get_model_result(pipe_xgb, 'xgb')

    result = pd.concat([_result_knn, _result_lr, _result_dt, _result_svc, _result_nb, _result_rf, _result_xgb], axis = 1)

    print(result.round(4))
    return (
        pipe_dt,
        pipe_knn,
        pipe_lr,
        pipe_nb,
        pipe_rf,
        pipe_svc,
        pipe_xgb,
        result,
    )


@app.cell
def _():
    # Import Scorer
    from sklearn.metrics import make_scorer

    # Import Scores
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import matthews_corrcoef

    from imblearn.metrics import geometric_mean_score
    return (
        accuracy_score,
        f1_score,
        geometric_mean_score,
        make_scorer,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Performance Measures""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""So far, we have only used accuracy as our metric. However, accuracy does not portray the entire performance spectrum. For instance, what is the accuracy rate for positive patients? How accurate the model was in correctly predicting diabetes?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To get answers to these questions, there are other metrics that need to be considered. These metrics are defined using the confusion matrix. The image below represents such measures.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![confusion matrix.jpg](https://www.researchgate.net/publication/387699576/figure/fig4/AS:11431281301332250@1735959677437/Confusion-matrix-and-performance-indicator.png)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For classification and regression problems, there are separate performance measures. Details will be discussed in your class.

        You can get more idea from these medium articles: https://medium.com/analytics-vidhya/complete-guide-to-machine-learning-evaluation-metrics-615c2864d916
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **For classification** -

        *   Sensitivity/ Recall/ TPR
        *   Specificity/ TNR
        *   Precision
        *   PPV
        *   NPV

        **Compound metrics** (more robust measure of performance) -

        *   Balanced accuracy
        *   Geometric mean
        *   F-score
        *   MCC
        *   ROC-AUC
        """
    )
    return


@app.cell
def _(
    accuracy_score,
    geometric_mean_score,
    make_scorer,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
):
    scores={'accuracy': make_scorer(accuracy_score),
            'recall' : make_scorer(recall_score),
            'specificity':make_scorer(recall_score,pos_label=0),
            'gmean': make_scorer(geometric_mean_score),    # is available in imblearn library, not in sklearn
            'roc': make_scorer(roc_auc_score),
            'mcc': make_scorer(matthews_corrcoef)
            }
    return (scores,)


@app.cell
def _(
    get_model_result,
    pd,
    pipe_dt,
    pipe_knn,
    pipe_lr,
    pipe_nb,
    pipe_rf,
    pipe_svc,
    pipe_xgb,
    scores,
):
    _result_knn = get_model_result(pipe_knn, 'knn', scoring=scores)

    _result_lr = get_model_result(pipe_lr, 'lr', scoring=scores)

    _result_dt = get_model_result(pipe_dt, 'dt', scoring=scores)

    _result_svc = get_model_result(pipe_svc, 'svc', scoring=scores)

    _result_nb = get_model_result(pipe_nb, 'nb', scoring=scores)

    _result_rf = get_model_result(pipe_rf, 'rf', scoring=scores)

    _result_xgb = get_model_result(pipe_xgb, 'xgb', scoring=scores)

    result_ex = pd.concat([_result_knn, _result_lr, _result_dt, _result_svc, _result_nb, _result_rf, _result_xgb], axis = 1)

    print(result_ex.round(4))
    return (result_ex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As you can see, the recall score is pretty low, around 60% - this is undesirable. The MCC score is also very poor. It is important to correctly identify postive cases.

        The classifier prediction is biased towards the negative cases (majority class). Metrics such as accuracy do not portray that.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Cost-sensitive learning""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One way to deal with the class imbalance scenario. Here, the cost function is modified to prioritize the the positive instances. Details will be discussed in the class.

        Although the data has a small imbalance, you can still see how to perform this operation.

        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        look at the documentation for the SVM classifier. There is a parameter termed 'class_weight'.
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
