# RecurrentForest

This repo is a rapidly prototyped idea I had after re-reading some materials from my neuroscience days. Numenta's 1000 brains theory of intelligence involves cortical columns that each form a model of an object based on different sensory input. Long-range connections between columns gives each column access to information from other columns. This reminded me of ensemble methods, so I coded up a tree algorithm based on this idea. 

This model is somewhere between bagging and boosting. Each tree gets a subset of features from the training data dictated by a random vector theta, as in RandomForest. This is similar to how cortical columns might get different sensor information. In my model (called RecurrentForest) each tree is randomly coupled to other trees, emulating the way a single cortical column receives sparse long range connections from other columns. Each tree gets to use the predictions of trees coupled to it. This is simmilar to boosting. Training is as follows

For time=1:T

for tree in forest

tree.train(X, theta)

predictions = tree.predict()

tree.X <- tree.X UNION predictions from other trees this tree is coupled to

So the algoritihm basically trains each tree, updates features, and repeats.
    
Preliminarily, it appears the algorithm can get high test set accuracy on toy problems (same ballpark as RandomForest or AdaBoost), but the results appear to depend heavily on parameters like n_trees. Investigation ongoing...

A fun idea going forward would be to learn a useful coupling between trees based on some kind of plasticity rule. 
