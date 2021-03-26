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


rfc.train()

treewise_acc_local = np.zeros((n_trees, T))
treewise_acc_global = np.zeros((n_trees, T))
treewise_acc_test = np.zeros((n_trees, T))

for n in range(n_trees):
    for t in range(T):
        y_hat_l, y_l = rfc.trees[t][n]._get_local_train_preds()
        y_hat_g, y_g = rfc.trees[t][n]._get_overall_train_preds()
        y_hat_t = rfc.trees[t][n].predict(X_test, t, sparse_n=False)
        local_acc = np.mean(y_hat_l == y_l)
        global_acc = np.mean(y_hat_g == y_g)
        test_acc = np.mean(y_hat_t == y_test)
        treewise_acc_local[n,t] = local_acc
        treewise_acc_global[n,t] = global_acc
        treewise_acc_test[n,t] = test_acc

fig, ax = plt.subplots(3)
ax[0].set_title("local train acc")
ax[1].set_title("global train acc")
ax[2].set_title("global test acc")

for i in range(n_trees):
    print(f"input to tree {i}: {rfc.trees[0][i].sparse_adj}")
    ax[0].plot(treewise_acc_local[i,:], label=f'{i}')
    ax[1].plot(treewise_acc_global[i, :], label=f'{i}')
    ax[2].plot(treewise_acc_test[i,:], label=f"{i}")
    
plt.legend()
# plt.tight_layout()
plt.show()
    

# # get training accuracy of each tree
# print("training scores: t=0")
# for tr in rfc.trees[0]:
#     preds = tr.predict(rfc.X_train, 0, sparse_n=True, sparse_d=True)
#     ys = rfc.y_train[tr.sparse_n]
#     print(f"training accuracy: {np.mean(ys == preds)}")


# # get the overall accuracy of each tree
# print("\n\noverall training scores: t=0")
# for tr in rfc.trees[0]:
#     preds = tr.predict(rfc.X_train, 0, sparse_n=False, sparse_d=True)
#     print(f"training accuracy: {np.mean(rfc.y_train == preds)}")


# # get training accuracy of each tree at the first time step
# print("\n\ntraining scores: t=1")
# for tr in rfc.trees[1]:
#     preds = tr.predict(rfc.X_train, 1, sparse_n=True, sparse_d=True)
#     ys = rfc.y_train[tr.sparse_n]
#     print(f"training accuracy: {np.mean(ys == preds)}")


# # get the overall accuracy of each tree
# print("\n\noverall training scores: t=1")
# for tr in rfc.trees[1]:
#     preds = tr.predict(rfc.X_train, 1, sparse_n=False, sparse_d=True)
#     print(f"training accuracy: {np.mean(rfc.y_train == preds)}")
