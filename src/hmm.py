from util import generate_graph
import numpy as np
import math


class HMM():
    def __init__(self, hidden_layers) -> None:
        self.hidden_layers = hidden_layers
        self.stat_distr = self.__calculateStationaryDistribution()
        self.prev_X = None
        self.prob_arr = self.stat_distr.copy()
    
    @staticmethod
    def build(connectivityMatrix, distribution='linear', mu=0, sigma=2, max_dist=11):
        dm = createDistanceMatrix(connectivityMatrix)
        if distribution == 'linear':
            matrix = createLinearDistributionMatrix(dm)
        elif distribution == 'gaussian':
            matrix = createGaussianDistributionMatrix(dm, mu, sigma, max_dist)
        return HMM(matrix)
    
    def getOptimalPrediction(self, frame_room_prob, viterbi="False"):
        ## Return False if list isn't same size
        if len(frame_room_prob) != len(self.hidden_layers):
            return False
        
        if viterbi:        
            best_pred = self.__viterbi(frame_room_prob)
        else:
            best_pred = self.__predictWithoutSequence(frame_room_prob)
        return best_pred
    
    def __calculateStationaryDistribution(self):
        matrix_transposed = np.array(self.hidden_layers).T
        eigenvals, eigenvects = np.linalg.eig(matrix_transposed)
        idx = np.isclose(eigenvals, 1)
        target_eigenvect = eigenvects[:, idx]
        target_eigenvect = target_eigenvect[:, 0]
        stat_distr = target_eigenvect / sum(target_eigenvect)
        return stat_distr

    def __predictWithoutSequence(self, frame_room_prob):
        best_pred = (0, None)
        prev_X_list = []
        ## Use stationary distribution on first run
        if self.prev_X == None:
            prev_X_list = self.stat_distr
        else:
            prev_X_list = self.hidden_layers[self.prev_X]

        ## Calculate the optimal prediction
        for i, room_prob in enumerate(frame_room_prob):
            pred = prev_X_list[i] * room_prob
            if pred > best_pred[0]:
                best_pred = (pred, i)
        self.prev_X = best_pred[1]
        return best_pred
    
    ## viterbi algorithm (kinda)
    def __viterbi(self, room_prob):
        global_max = (0, None)
        for i, prev_prob in enumerate(self.prob_arr):
            local_max = (0, None)
            for j, p in enumerate(room_prob):
                next_p = prev_prob * self.hidden_layers[i][j] * p
                if next_p > local_max[0]:
                    local_max = (next_p, j)
            self.prob_arr[i] = local_max[0]
            if local_max[0] > global_max[0]:
                global_max = local_max
        return global_max               


### -- End class -- ##

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

def createLinearDistributionMatrix(distanceMatrix):
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

def getGaussianDistribution(mu, sigma, max):
    distr = []
    for i in range(0, max):
        z = (i - mu)/sigma
        distr.append(math.exp(z*z*-1/2)/math.sqrt(2*math.pi))
    return distr


def createGaussianDistributionMatrix(distanceMatrix, mu=0, sigma=1, max_dist=15):
    gauss_distr = getGaussianDistribution(mu, sigma, max_dist)
    wm = [row.copy() for row in distanceMatrix]
    for row in wm:
        dist_max = max(row)
        sum = 0
        for i, dist in enumerate(row):
            new_val = (dist_max - dist + 1) * gauss_distr[int(dist)]
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
    wm = createLinearDistributionMatrix(dm)
    gm = createGaussianDistributionMatrix(dm)

    """
    print("Connectivity Matrix:")
    printMatrix(cm)
    print("\n\nDistance Matrix:")
    printMatrix(dm)
    print("\n\nWeightedMatrix (= hidden layers): ")
    printMatrix(wm)
    print(sum(wm[0]))
    """
    printMatrix(wm)
    print("\n\n\n")
    printMatrix(gm)

#gaussian_kernel = cv.getGaussianKernel(length)



