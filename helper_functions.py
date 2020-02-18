import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

def train_test_split_df (df,test_size = 0.2):
    '''Function that helps to split the data as train set or test set'''
    idx = list(df.index)
    random.Random(seed).shuffle(idx)
    test_len = int(test_size * len(df))
    return df.loc[idx[test_len:]],df.loc[idx[:test_len]]

def model_fit_train_score(model, x_train, y_train, x_val, y_val):
    '''This function takes in five arguments:model (model object), X_train, y_train, X_test_val,y_val
    The data will be fitted using the model passed in by the user
    It returns the fitted model object and Accuracy score as well as F1 score and AUC (area under curve)'''
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    predictions = model.predict_proba(x_val)

    keys = ['predictions', 'predict_proba', 'Accuracy', 'F1', 'AUC']

    results_dict = dict.fromkeys(keys)
    results_dict['predictions'] = y_pred
    results_dict['predict_proba'] = predictions[:, 1]
    results_dict['Accuracy'] = accuracy_score(y_true=y_val, y_pred=y_pred)
    results_dict['F1'] = f1_score(y_true=y_val, y_pred=y_pred)
    results_dict['AUC'] = roc_auc_score(y_val, predictions[:, 1])

    return model, results_dict


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=10):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names, )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Greens')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def plot_ROC(y_true,y_proba,AUC,figsize = (7,5),color = 'darkturquoise',title='ROC Curve'):
    '''Helper function to plot ROC graph'''
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    #set size
    plt.figure(figsize=figsize)
    #plot
    plt.plot(fpr, tpr,lw=2,c=color,label = f"AUC: {AUC:.2f}")
    #adjustments
    plt.plot([0,1],[0,1],c='grey',ls='--')
    plt.legend()
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title);


def model_fit_train_score_skf(model, X, y, kfold=5):
    '''This function takes in three arguments:model (model object), X,y
    It will be splitted by stratified k fold algo
    The data will be fitted using the model passed in by the user
    It returns the fitted model object and lists of Accuracy score as well as F1 score and AUC (area under curve)'''
    skf = StratifiedKFold(n_splits=kfold)
    results_dict = defaultdict()
    predict = []
    predict_prob = []
    Accuracy = []
    F1 = []
    AUC = []
    y_vals = []

    for train_index, test_index in skf.split(X, y):
        # get current split
        x_train, x_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        # fit model with latest train set
        model.fit(x_train, y_train)
        # calculate predictions
        y_pred = model.predict(x_val)
        predictions = model.predict_proba(x_val)
        y_vals.append(y_val)
        predict.append(y_pred)
        predict_prob.append(predictions[:, 1])
        Accuracy.append(accuracy_score(y_true=y_val, y_pred=y_pred))
        F1.append(f1_score(y_true=y_val, y_pred=y_pred))
        AUC.append(roc_auc_score(y_val, predictions[:, 1]))

    results_dict['y_val'] = y_vals
    results_dict['predictions'] = predict
    results_dict['predict_proba'] = predict_prob
    results_dict['Accuracy_mean'] = np.mean(Accuracy)
    results_dict['F1_mean'] = np.mean(F1)
    results_dict['AUC_mean'] = np.mean(AUC)
    results_dict['Accuracy_std'] = np.std(Accuracy)
    results_dict['F1_std'] = np.std(F1)
    results_dict['AUC_std'] = np.std(AUC)

    return model, results_dict

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)

def lemma_and_stop (text):
    docs = nlp.pipe(text)
    lemma = []
    lemma_and_stop = []

    for doc in docs:
        if doc.is_parsed:
            lemma.append(" ".join([n.lemma_ for n in doc]))
            lemma_and_stop.append(" ".join([n.lemma_ for n in doc if n.is_stop == False]))
        else:
            # We want to make sure that the lists of parsed results have the
            # same number of entries of the original Dataframe, so add some blanks in case the parse fails
            lemma.append(None)
            lemma_and_stop.append(None)
    return lemma,lemma_and_stop