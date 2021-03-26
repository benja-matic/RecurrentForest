from sklearn.model_selection import train_test_split as TTS
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import make_classification

import numpy as np
from tqdm import tqdm
from scipy.special import softmax as softmax
import matplotlib.pyplot as plt

import importlib
import RecurrentForest as RF
from Utils import FileIO as FIO
from Utils import Metrics as M


def singleExperiment(cfg):
    """
    Can only run with access to config variable
    """
    # make data
    X, y = make_classification(n_samples=cfg['n_samples'],
                               n_features=cfg['n_features'])
    # train test split
    X_train, X_test, y_train, y_test = TTS(X, y, test_size=.2)
    # set up RecurrentForest model
    rec_fst_clf = RF.RecurrentForest(X_train,
                             y_train,
                             cfg['T'],
                             cfg['n_trees'],
                             cfg['p_connect'],
                             cfg['p_feature'],
                             cfg['p_example'],
                             cfg['tree_kwargs'])
    # set up RandomForest
    rnd_fst_clf = RFC(**cfg['random_forest_kwargs'])
    # set up AdaBoost
    ada_bst_clf = ABC(**cfg['ada_boost_kwargs'])
    # in a list
    models = [rec_fst_clf,
              rnd_fst_clf,
              ada_bst_clf]
    
    print("<<< training models >>>")
    for m in tqdm(models):
        m.fit(X_train, y_train) # RecurrentForest ignores args - data present at init

    print("<<< testing models >>>")
    y_hats = np.zeros((3, X_test.shape[0]))
    for i, m in tqdm(enumerate(models)):
        if i == 0:
            y_hats[i, :] = m.predictNew(X_test)
        else:
            y_hats[i, :] = m.predict(X_test)

    # get metrics
    measures = np.zeros((3, 4))
    for i in tqdm(range(3)):
        measures[i,:] = M.binary_metrics(y_test, y_hats[i,:], model=str(models[i]))

    return measures


def batch_experiment(cfg):

    measures = np.zeros((3,4,cfg['n_runs']))
    for i in range(cfg['n_runs']):
        measures[:,:,i] = singleExperiment(cfg)

    return measures


if __name__ == '__main__':
    cfg = FIO.load_cfg("make_classification_config.yaml")
    measures = batch_experiment(cfg)
    info_string = "3x4xn_runs: \n\tRecurrentForest, RandomForest, AdaBoost\n\tAccuracy, Precision, Recall, F1"
    FIO.save_exp([measures, info_string])
    

