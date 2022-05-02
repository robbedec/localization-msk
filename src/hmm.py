#from hmmlearn.hmm import GaussianHMM
import enum
from graph import Graph
from util import generate_graph
from scipy.stats import norm
import numpy as np
import cv2 as cv


class HMM():
    def __init__(self, hidden_layers) -> None:
        self.hidden_layers = hidden_layers
        self.stat_distr = self.__calculateStationaryDistribution()
        self.currentOdds = 1
        self.prev_X = None
    
    @staticmethod
    def build(connectivityMatrix):
        dm = createDistanceMatrix(connectivityMatrix)
        wm = createWeightedMatrix(dm)
        return HMM(wm)
    
    def getOptimalPrediction(self, frame_room_prob):
        ## Return False if list isn't same size
        if len(frame_room_prob) != len(self.hidden_layers):
            return False

        
        best_pred = (0, None)
        prev_X_list = []
        ## Use stationary distribution on first run
        if self.prev_X == None:
            prev_X_list = self.stat_distr
        else:
            prev_X_list = self.hidden_layers[self.prev_X]

        ## Calculate the optimal prediction
        for i, room_prob in enumerate(frame_room_prob):
            pred = self.__calculateOdds(self.currentOdds, prev_X_list[i], room_prob)
            if pred > best_pred[0]:
                best_pred = (pred, i)
        self.prev_X = best_pred[1]
        return best_pred
    
    def __calculateStationaryDistribution(self):
        matrix_transposed = np.array(self.hidden_layers).T
        eigenvals, eigenvects = np.linalg.eig(matrix_transposed)
        idx = np.isclose(eigenvals, 1)
        target_eigenvect = eigenvects[:, idx]
        target_eigenvect = target_eigenvect[:, 0]
        stat_distr = target_eigenvect / sum(target_eigenvect)
        return stat_distr

    def __calculateOdds(self, current_odds, P_X, P_YX):
        return current_odds * P_X * P_YX
    
        """
        print(obs_var_matrix)
        for row in obs_var_matrix:
            contour_idx = []
            for i, val in enumerate(row):
                if val != 0:
                    contour_idx.append(i)
            idx_list.append(contour_idx)
        stat_distr = getStationaryDistribution(hidden_states)
        print(idx_list)
        idx_combinations = [list(itertools.combinations_with_replacement(row, r=len(row))) for row in idx_list]
        idx_cart_product = [list(itertools.product(row, repeat=len(row))) for row in idx_list]

        print(idx_combinations)
        print(idx_cart_product)
        """
def createDistanceMatrix(connectivityMatrix):
    length = len(connectivityMatrix)
    dm = [row.copy() for row in connectivityMatrix]
    for k in range(0, length):
        for i in range(0, length):
            for j in range(0, length):
                if ((i != k) & (dm[i][k] == 0)) | ((k != j) & (dm[k][j] == 0)) | (i == j) :
                    continue
                target = dm[i][j]
                if target == 0:
                    target = 999
                if dm[i][k] + dm[k][j] < target:
                    dm[i][j] = dm[i][k] + dm[k][j]
    return dm

def createWeightedMatrix(distanceMatrix):
    wm = [row.copy() for row in distanceMatrix]
    for row in wm:
        dist_max = max(row)
        sum = 0
        for i, dist in enumerate(row):
            new_val = dist_max - dist + 1
            row[i] = new_val
            sum += new_val
        row /= sum
    return wm

def printMatrix(matrix):
    for row in matrix:
        print(row)
    
if __name__ == '__main__':
    g = generate_graph()
    cm = g.getConnectivityMatrix()
    dm = createDistanceMatrix(cm)
    wm = createWeightedMatrix(dm)


    print("Connectivity Matrix:")
    printMatrix(cm)
    print("\n\nDistance Matrix:")
    printMatrix(dm)
    print("\n\nWeightedMatrix (= hidden layers): ")
    printMatrix(wm)
    print(sum(wm[0]))

#gaussian_kernel = cv.getGaussianKernel(length)



