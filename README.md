# RecurrentForest

This repo is a rapidly prototyped idea I had after re-reading some materials from my neuroscience days. Numenta's 1000 brains theory of intelligence involves cortical columns that each form a model of an object based on different sensory input. Long-range connections between columns gives each column access to information from other columns. This reminded me of ensemble methods, so I coded up a tree algorithm based on this idea. 

This model is somewhere between bagging and boosting. Each tree gets a subset of features from the training data dictated by a random vector theta, as in RandomForest, similar to how cortical columns might get different sensor information. Each tree is randomly coupled to other trees though, and gets to use other trees predictions as features. This is simmilar to boosting. Training is as follows

For time=1:T
  for tree in forest
    tree.train(X, theta)
    predictions = tree.predict()
    tree.X <- tree.X UNION predictions from other trees this tree is coupled to
    
Preliminarily I tested this approach on 20 randomly generated classification problems and it peformed somewhere in between randomforest and adaboost in terms of accuracy. There is probably a theoretical reason that my model is just doing the same thing as RandomForest asymptotically, but the idea here was to code something up inspired by neuroscience. 
    
    
