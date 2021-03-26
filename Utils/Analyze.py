import numpy as np
import matplotlib.pyplot as plt

class ExpResults:

    """
    Skeleton code, not yet used
    """

    def __init__(self, measures, info):

        self.info = info
        self.measures = measures

    def get_means_runwise(self):

        self.means = np.mean(self.measures, axis=2))
        return self.means

    def get_stds_runwise(self):

        self.stds = np.std(self.measures, axis=2)
        return self.stds

    def quad_plot(self):

        print("not yet implemented")
        


