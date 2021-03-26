import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from tqdm import tqdm
from scipy.special import softmax as softmax


def GetDataset(X, y, rows, cols):
    return X[rows, cols], y[rows]


class RecurrentForest:

    def __init__(self, X, y, T, n_trees, p_connect, p_feature, p_example, tree_kwargs):

        # Data for lightweight pass by reference
        self.T = T  # number of rounds
        self.X = X  # training data
        self.y = y  # labels
        self.n_classes = len(np.unique(self.y))
        self.n_trees = n_trees  # number of trees in forest
        n, d = self.X.shape
        self.n = n  # number of training examples
        self.d = d  # number of features in each example
        self.tails = np.zeros((self.n, self.n_trees))
        self.tree_kwargs = {} if not tree_kwargs else tree_kwargs

        # Probabilities
        probs = [p_connect, p_feature, p_example]
        self.p_connect = p_connect  # p(coupling to each other tree)
        self.p_feature = p_feature  # p(using a feature)
        self.p_example = p_example  # p(drawing an example)

        assert sum([(i >= 0) & (i < 1) for i in probs]) == 3, "please ensure the last three arguments\
                                                                are all valid probabilities"

        # coupling between trees
        self.Adjacency = np.random.uniform(
            0, 1, (n_trees, n_trees)) < p_connect
        # subsets of features
        self.Features = np.random.uniform(0, 1, (n_trees, d)) < p_feature
        # subsets of training examples
        self.Examples = np.random.uniform(0, 1, (n_trees, n)) < p_example

        self.trees = [[None for i in range(self.n_trees)] for j in range(T)]

        for i in range(n_trees):
            sparse_adj = np.where(self.Adjacency[i, :])[0]
            sparse_features = np.where(self.Features[i, :])[0]
            sparse_examples = np.where(self.Examples[i, :])[0]
            Tree = RandomTree(self.X,
                              self.y,
                              self.tails,
                              sparse_examples,
                              sparse_features,
                              sparse_adj,
                              0,
                              i,
                              tree_kwargs = self.tree_kwargs)
            self.trees[0][i] = Tree

        print("...Forest Initialized...")

    def __str__(self):
        return f"RecurrentForest(T={self.T}, p_connect={self.p_connect}, p_feature={self.p_feature}, p_example={self.p_example})"

    def initializeTree(self, time, idx):
        sparse_vectors = self.trees[0][idx].getSparseVectors()
        tree = RandomTree(self.X, self.y, self.tails,
                            *sparse_vectors, t=time, idx=idx,
                            tree_kwargs=self.tree_kwargs)
        self.trees[time][idx] = tree

    def updateTails(self, time):
        for tree in range(self.n_trees):
            self.tails[:, tree] = self.trees[time][tree].predictTrain(time)

    def updateNewTails(self, time, newX):
        for tree in range(self.n_trees):
            self.newTails[:, tree] = self.trees[time][tree].predictNew(
                newX, time)

    def setNewTails(self, time):
        for tree in range(self.n_trees):
            self.trees[time][tree].newTails = self.newTails

    def train(self):
        print("<<< Training RecurrentForest >>>")
        for time in tqdm(range(self.T)):

            for tree in range(self.n_trees):
                    self.trees[time][tree].train(time)

            if time < self.T - 1:
                self.updateTails(time)
                for j in range(self.n_trees):
                    self.initializeTree(time+1, j)

    def fit(self, X, y):
        """
        Alias for sklearn consistency
        """
        self.train()

    def predictProbaNew(self, X):

        self.newTails = np.zeros((X.shape[0], self.n_trees))
        for time in range(self.T):
            self.setNewTails(time)
            self.updateNewTails(time, X)

        # now newTails is n x trees with predictions
        frequencies = np.zeros((X.shape[0], self.n_classes))
        for val in range(self.n_classes):
            bool_table = self.newTails == val
            frequencies[:, val] = np.sum(bool_table, axis=1)

        probs = softmax(frequencies, axis=1)
        return probs

    def predictNew(self, X):
        probs = self.predictProbaNew(X)
        return np.argmax(probs, axis=1)


class RandomTree:

    """

    """

    def __init__(self, X, y, tails, sparse_n, sparse_d, sparse_adj, t=None, idx=None, tree_kwargs=None):
        """
        A decision tree with some methods

        Arguments
            X_train: np.ndarray  --  full dataset passed by reference from RecurrentForest class
            y_train : np.ndarray  --  y values for full dataset, works just like X
            train_tails: np.ndarray  --  0s initially, bulletin board for output of trees
                                    some subset of this gets added to the 'tail' of a
                                    given tree's X
            sparse_n: np.ndarray  --  mask for training examples into X
            sparse_d: np.ndarray  --  sparse_n, except features not examples
            sparse_adj: np.ndarray  --  same, but tells which trees I receive input from
            t: int  --  time step in training process
            idx: int  --  index in list of trees in forest
            tree_kwargs: dict  --  arguments like max_depth etc. for decision tree
        """

        # attributes use pass by reference
        # so self.X acts like a pointer to RecurrentForest.X
        self.X = X
        self.y = y
        self.tails = tails  # predictions from previous trees
        self.sparse_n = sparse_n  # which training examples
        self.sparse_d = sparse_d  # which features
        self.sparse_adj = sparse_adj  # which trees to I get input from
        self.t = 0 if t is None else t  # which round in {1..T} am I on
        self.idx = idx  # my position in RecurrentForest.trees[t]
        self.tree_kwargs = {} if tree_kwargs is None else tree_kwargs

    def getSparseVectors(self):
        return [self.sparse_n, self.sparse_d, self.sparse_adj]

    def getFirstX(self):
        return self.X[self.sparse_n, :][:, self.sparse_d]

    def getFirstXDense(self):
        return self.X[:, self.sparse_d]

    def getFirstXNew(self, newX):
        return newX[:, self.sparse_d]

    def getX(self):
        return np.hstack([self.X[self.sparse_n, :][:, self.sparse_d], self.tails[self.sparse_n, :][:, self.sparse_adj]])

    def getXDense(self):
        return np.hstack([self.X[:, self.sparse_d], self.tails[:, self.sparse_adj]])

    def getXNew(self, newX):
        return np.hstack([newX[:, self.sparse_d], self.newTails[:, self.sparse_adj]])

    def train(self, time):
        self.decTree = DTC(**self.tree_kwargs)
        if time == 0:
            self.decTree.fit(self.getFirstX(), self.y[self.sparse_n])
        else:
            self.decTree.fit(self.getX(), self.y[self.sparse_n])

    def predictTrain(self, time):
        if time == 0:
            y_hat_dense = self.decTree.predict(self.getFirstXDense())
        else:
            y_hat_dense = self.decTree.predict(self.getXDense())
        self.y_hat_dense = y_hat_dense
        return y_hat_dense

    def predictNew(self, newX, time):
        if time == 0:
            y_hat_new = self.decTree.predict(self.getFirstXNew(newX))
        else:
            y_hat_new = self.decTree.predict(self.getXNew(newX))
        self.y_hat_new = y_hat_new
        return y_hat_new
