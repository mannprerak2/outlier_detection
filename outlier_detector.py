import random
from statistics import mean, stdev
from typing import Callable
from time import time

class DistanceHandler:
    '''
    Handles Distance calculations including caching.

    - data: array of data
    - calculate: function to calculate distance between 2 points.
        - Input: data point 1 & 2
        - Output: distance value
    '''

    def __init__(self, data, calculate: Callable):
        self.data = data
        self.cache = {}
        self.calculate = calculate

    def distance(self, i, j):
        if i > j:
            return self.distance(j, i)

        key = "{}_{}".format(i, j)
        if key not in self.cache:
            self.cache[key] = self.calculate(self.data[i], self.data[j])
        return self.cache[key]

    def closest(self, i, arr) -> int:
        '''
        Finds the point closest to [i] in [arr].
        '''
        distances = [self.distance(i, x) for x in arr]
        return arr[distances.index(min(distances))]


class KNNValues:
    '''
    Stores the k nearest neighbours of a data item.
    kNN: Number of k nearest neigbours
    distances: sorted (ascending) distances to k nearest neighbours.
    '''

    def __init__(self, kNN, distances):
        self.kNN = kNN
        self.__average = None
        self.distances = distances
        self.distances.sort()

    def getMax(self):
        return self.distances[-1]

    def setMax(self, value):
        '''
        Sets the max value to [value] and sorts [distances].
        '''
        self.distances[-1] = value
        self.__average = None
        self.distances.sort()

    def average(self):
        if self.__average is None:
            self.__average = mean(self.distances)
        return self.__average


class Node:
    '''
    A cluster.
    '''

    def __init__(self, sampleIndexes=[], verifyCenters=False, centroidSeeds=[]):
        # Indices of all samples under this cluster.
        self.sampleNumbers = []
        self.sampleNumbers.extend(sampleIndexes)
        self.verifyCenters = verifyCenters
        self.centroidSeeds = centroidSeeds

    def addSample(self, sample):
        self.sampleNumbers.append(sample)

    def __update_dist_and_edge_knn(self, i, j, distance, dist_knn, edge_knn):
        # For i.
        if dist_knn[i].getMax() > distance.distance(i, j):
            dist_knn[i].setMax(distance.distance(i, j))
            edge_knn[i] = dist_knn[i].average()
        # For j.
        if dist_knn[j].getMax() > distance.distance(i, j):
            dist_knn[j].setMax(distance.distance(i, j))
            edge_knn[j] = dist_knn[j].average()

    def DHCA(self, dist_knn, edge_knn, kNN, nodeArray,
             k, maxClusterSize, threshold,
             distance: DistanceHandler, verifiedStatus) -> None:
        '''
        Input:
        - dist_knn: sorted distances of kNN for each data item.
        - edge_knn: db outlier scores
        - kNN: number of kNNs for a data item.
        - nodeArray: Array of Nodes(clusters).
        - k: number of clusters at each step.
        - maxClusterSize: max size allowed for a cluster.
        - threshold: The value used to filter.

        Output: (Returns None)
        - Updated dist_knn and edge_knn (inplace).
        - New nodes generated (<=k) and pushed to nodeArray (inplace).
        '''

        if len(self.centroidSeeds) < 1:
            # Randomly select k elements from sampleNumbers
            self.centroidSeeds = random.sample(
                population=self.sampleNumbers, k=k)

        # Generate k new nodes and map them with the index.
        nodeMap = {}
        for center in self.centroidSeeds:
            nodeMap[center] = Node([center])

        for i in self.sampleNumbers:
            if i in self.centroidSeeds:
                continue
            # Loop over nodes that are not center.

            # Get nearest center.
            j = distance.closest(i, self.centroidSeeds)

            self.__update_dist_and_edge_knn(
                i, j, distance=distance, dist_knn=dist_knn, edge_knn=edge_knn)

            if dist_knn[i].average() > threshold:
                nodeMap[j].addSample(i)

        if self.verifyCenters:
            for c1 in range(0, len(self.centroidSeeds)):
                # Skip this if center is already verified.
                if verifiedStatus[self.centroidSeeds[c1]]:
                    continue
                for c2 in range(c1+1, len(self.centroidSeeds)):
                    self.__update_dist_and_edge_knn(
                        self.centroidSeeds[c1], self.centroidSeeds[c2],
                        distance=distance, dist_knn=dist_knn, edge_knn=edge_knn)
                verifiedStatus[self.centroidSeeds[c1]] = True

        for v in nodeMap.values():
            if len(v.sampleNumbers) > maxClusterSize:
                nodeArray.append(v)


class Result:
    '''
    Holds generated result returned by [Runner.run()]
    '''

    def __init__(self, outlier_indexes, verifiedStatus, calculations, runningTime):
        self.outlier_indexes = outlier_indexes
        self.verifiedStatus = verifiedStatus
        self.calculations = calculations
        self.runningTime = runningTime


class Runner:
    '''
    Handles running the algo.
    - kNN: number of kNNs for a data item.
    - k: number of clusters at each step.
    - n: number of outliers to detect.
    - maxClusterSize: max size allowed for a cluster.
    - data: array of data points
    - calculateDistance: a function to calculate distance between 2 data points.
    '''

    def __init__(self, kNN, k, n, maxClusterSize, data, calculateDistance):
        self.kNN = kNN
        self.k = k
        self.n = n
        self.maxClusterSize = maxClusterSize
        self.data = data
        self.distanceHandler = DistanceHandler(data, calculateDistance)

        '''
        - dist_knn: sorted distances of kNN for each data item.
        - edge_knn: db outlier scores
        - db_outlier_indexes: sorted (reverse) sample indexes on key=edge_knn
        '''

    def __sortOutliersDesc(self):
        self.db_outlier_indexes.sort(
            reverse=True, key=lambda x: self.edge_knn[x])

    def run(self) -> Result:
        '''
        Run the algorighm and return [Result].
        '''
        startTime = time()
        size = len(self.data)
        sampleIndexes = list(range(0, size))
        self.dist_knn = [0]*size
        self.edge_knn = [0]*size
        self.db_outlier_indexes = []

        # Array of verification status of all data items. True if verified.
        self.verifiedStatus = [False]*size

        # Fill sequential dist_knn values for initial upper bound.
        for i in range(0, size):
            distances = [0]*self.kNN
            for j in range(0, self.kNN):
                distances[j] = self.distanceHandler.distance(i, (i+j+1) % size)
            self.dist_knn[i] = KNNValues(self.kNN, distances)

        # Set initial edge_knn (db outlier score).
        for i in range(0, size):
            self.edge_knn[i] = self.dist_knn[i].average()

        # Sorted (desc) list of indexes.
        self.db_outlier_indexes = list(range(0, size))
        self.__sortOutliersDesc()

        while not self.areTopNVerified():
            # Use top K unverified as first level cluster centers.
            nodeArray = [Node(sampleIndexes=sampleIndexes,
                              centroidSeeds=self.getTopKUnverified(), verifyCenters=True)]

            threshold = self.getThreshold()

            for node in nodeArray:
                node.DHCA(dist_knn=self.dist_knn, edge_knn=self.edge_knn,
                          kNN=self.kNN, nodeArray=nodeArray, k=self.k, maxClusterSize=self.maxClusterSize,
                          distance=self.distanceHandler, threshold=threshold, verifiedStatus=self.verifiedStatus)

            # Sort db outliers
            self.__sortOutliersDesc()

        return Result(
            outlier_indexes=self.db_outlier_indexes[:self.n],
            verifiedStatus=self.verifiedStatus,
            calculations=len(self.distanceHandler.cache),
            runningTime=time()-startTime
        )

    def getThreshold(self):
        return mean(self.edge_knn)+stdev(self.edge_knn)

    def areTopNVerified(self):
        for i in self.db_outlier_indexes[:self.n]:
            if not self.verifiedStatus[i]:
                return False
        return True

    def getTopKUnverified(self):
        ans = []
        for i in self.db_outlier_indexes:
            if not self.verifiedStatus[i]:
                ans.append(i)
                if len(ans) >= self.k:
                    break
        return ans
