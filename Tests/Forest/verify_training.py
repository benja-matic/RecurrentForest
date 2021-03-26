import numpy as np
from sklearn.datasets import make_classification
import RecurrentForest as RF
from sklearn.model_selection import train_test_split as TTS


X, y = make_classification(n_samples = 10_000,
                           n_features = 100)

X_train, X_test, y_train, y_test = TTS(X, y, test_size=.2)

n_trees = 10
T = 10
p_connect = .2
p_feature = .2
p_example = .2
tree_kwargs = {"max_depth": 1}

rec_fst_clf = RF.RecurrentForest(X_train,
                                 y_train,
                                 T,
                                 n_trees,
                                 p_connect,
                                 p_feature,
                                 p_example,
                                 tree_kwargs=tree_kwargs)

rec_fst_clf.train()
y_hat = rec_fst_clf.predict(X_train)
print(np.mean(y_hat == y_train))

y_hat = rec_fst_clf.predict(X_test)
print(np.mean(y_hat == y_test))
