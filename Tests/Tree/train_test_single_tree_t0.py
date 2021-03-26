import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as TTS
import RecurrentForest as RF
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=10_000,
                           n_features=100)

X_train, X_test, y_train, y_test = TTS(X, y, test_size=.2)

n_trees = 10
T = 10
p_connect = .2
p_feature = .2
p_example = .2
tree_kwargs = {}

rfc = RF.RecurrentForest(X_train,
                         y_train,
                         T,
                         n_trees,
                         p_connect,
                         p_feature,
                         p_example,
                         tree_kwargs=tree_kwargs)

train_accs = np.zeros(n_trees)
test_accs = np.zeros(n_trees)
for i in range(n_trees):
    rfc.trees[0][i].train(0)
    y_hat_train, y_trn = rfc.trees[0][i]._get_overall_train_preds()
    y_hat_test = rfc.trees[0][i].predict(X_test, 0, sparse_n=False)
    train_accs[i] = np.mean(y_hat_train == y_trn)
    test_accs[i] = np.mean(y_hat_test == y_test)

fig, ax = plt.subplots()

width = .3
offset = .3
ax.bar(np.arange(T), train_accs, width=width)
ax.bar(np.arange(T)+offset, test_accs, width=width)
plt.show()