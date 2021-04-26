import random
from statistics import mean, stdev
from typing import Callable
from time import time
from math import comb
import json


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
    invertedDistanceMap: map of key-> item and value-> distance.
    '''

    def __init__(self, kNN, distanceMap: dict):
        self.kNN = kNN
        self.__average = None
        self.points = set(distanceMap.keys())
        self.distances = list(distanceMap.items())
        self.__sortDistances()

    def __sortDistances(self):
        self.distances.sort(key=lambda x: x[1])

    def getMax(self):
        return self.distances[-1][1]

    def isNotKnn(self, i):
        return i not in self.points

    def setMax(self, value, i):
        '''
        Sets the max value to [value] and sorts [distances].
        '''

        # Update max.
        self.points.remove(self.distances[-1][0])
        self.points.add(i)
        self.distances[-1] = (i, value)

        self.__average = None
        self.__sortDistances()

    def average(self):
        if self.__average is None:
            self.__average = tuple(map(mean, zip(*self.distances)))[1]
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
        if dist_knn[i].isNotKnn(j) and dist_knn[i].getMax() > distance.distance(i, j):
            dist_knn[i].setMax(distance.distance(i, j), j)
            edge_knn[i] = dist_knn[i].average()
        # For j.
        if dist_knn[j].isNotKnn(i) and dist_knn[j].getMax() > distance.distance(i, j):
            dist_knn[j].setMax(distance.distance(i, j), i)
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

        # Generate new nodes from centroidSeeds.
        nodeMap = {}
        for center in self.centroidSeeds:
            nodeMap[center] = Node(sampleIndexes=[center])

        for i in self.sampleNumbers:
            # Only loop over nodes that are not center.
            if i in self.centroidSeeds:
                continue

            # Get nearest center.
            j = distance.closest(i, self.centroidSeeds)

            self.__update_dist_and_edge_knn(
                i, j, distance=distance, dist_knn=dist_knn, edge_knn=edge_knn)

            if dist_knn[i].average() > threshold:
                nodeMap[j].addSample(i)

        if self.verifyCenters:
            for i in self.centroidSeeds:
                if verifiedStatus[i]:
                    continue
                for j in self.sampleNumbers:
                    if i == j:
                        continue
                    self.__update_dist_and_edge_knn(
                        i, j, distance=distance, dist_knn=dist_knn, edge_knn=edge_knn)
                verifiedStatus[i] = True

        for v in nodeMap.values():
            if len(v.sampleNumbers) > maxClusterSize:
                nodeArray.append(v)


class Result:
    '''
    Holds generated result returned by [Runner.run()]
    '''

    def __init__(self, outlier_indexes, outlier_scores, verifiedCount,
                 calculations, runningTime,
                 n, k, kNN, maxClusterSize, dataSize):
        self.outlier_indexes = outlier_indexes
        self.outlier_scores = outlier_scores
        self.verifiedCount = verifiedCount
        self.calculations = calculations
        self.runningTime = runningTime
        self.n = n
        self.k = k
        self.kNN = kNN
        self.maxClusterSize = maxClusterSize
        self.dataSize = dataSize

        self.verifiedPercentage = '({:.2f}%)'.format(
            100*self.verifiedCount/self.dataSize)
        self.calculationPercentage = '({:.2f}%)'.format(
            100*self.calculations/comb(self.dataSize, 2))

    def toJSON(self, indent=None):
        return json.dumps(self.__dict__, indent=indent)


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
        '''
        - dist_knn: sorted distances of kNN for each data item.
        - edge_knn: db outlier scores
        - db_outlier_indexes: sorted (reverse) sample indexes on key=edge_knn
        '''
        self.dist_knn = [0]*size
        self.edge_knn = [0]*size
        self.db_outlier_indexes = list(range(0, size))

        # Array of verification status of all data items. True if verified.
        self.verifiedStatus = [False]*size

        # Fill sequential dist_knn values for initial upper bound.
        for i in range(0, size):
            distanceMap = {}
            for j in range(0, self.kNN):
                knnitem = (i+j+1) % size
                distanceMap[knnitem] = self.distanceHandler.distance(
                    i, knnitem)
            self.dist_knn[i] = KNNValues(self.kNN, distanceMap)

        # Set initial edge_knn (db outlier score).
        for i in range(0, size):
            self.edge_knn[i] = self.dist_knn[i].average()

        # Sorted outlier indexes.
        self.__sortOutliersDesc()

        while not self.areTopNVerified():
            # Use top K unverified as first level cluster centers.
            topKUnverified = self.getTopKUnverified()
            nodeArray = [Node(sampleIndexes=sampleIndexes,
                              centroidSeeds=topKUnverified, verifyCenters=True)]

            threshold = self.getThreshold()

            for node in nodeArray:
                node.DHCA(dist_knn=self.dist_knn, edge_knn=self.edge_knn,
                          kNN=self.kNN, nodeArray=nodeArray, k=self.k, maxClusterSize=self.maxClusterSize,
                          distance=self.distanceHandler, threshold=threshold, verifiedStatus=self.verifiedStatus)

            # Sort db outliers
            self.__sortOutliersDesc()

        return Result(
            outlier_indexes=self.db_outlier_indexes[:self.n],
            outlier_scores=[self.edge_knn[x]
                            for x in self.db_outlier_indexes[:self.n]],
            verifiedCount=sum(self.verifiedStatus),
            calculations=len(self.distanceHandler.cache),
            runningTime=time()-startTime,
            n=self.n,
            k=self.k,
            kNN=self.kNN,
            maxClusterSize=self.maxClusterSize,
            dataSize=size
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
