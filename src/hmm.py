#from hmmlearn.hmm import GaussianHMM
from graph import Graph
from util import generate_graph
from scipy.stats import norm
import numpy as np
import cv2 as cv




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

def getHiddenStates():
    g = generate_graph()
    cm = g.getConnectivityMatrix()
    dm = createDistanceMatrix(cm)
    wm = createWeightedMatrix(dm)
    return wm

def getStationaryDistribution(hidden_states):
    matrix_transposed = np.array(hidden_states).T
    eigenvals, eigenvects = np.linalg.eig(matrix_transposed)

    idx = np.isclose(eigenvals, 1)
    target_eigenvect = eigenvects[:, idx]
    target_eigenvect = target_eigenvect[:, 0]
    stat_distr = target_eigenvect / sum(target_eigenvect)
    return stat_distr

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
    print("\n\nWeightedMatrix: ")
    printMatrix(wm)
    print(sum(wm[0]))

#gaussian_kernel = cv.getGaussianKernel(length)



