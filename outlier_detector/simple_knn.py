import random
from statistics import mean, stdev
from typing import Callable
from time import time
import json

# Reuse these classes from dhca detector
from outlier_detector.dhca import DistanceHandler, KNNValues


class Result:
    '''
    Holds generated result returned by [Runner.run()]
    '''

    def __init__(self, outlier_indexes, outlier_scores, n, kNN, calculations, runningTime, dataSize):
        self.outlier_indexes = outlier_indexes
        self.outlier_scores = outlier_scores
        self.calculations = calculations
        self.runningTime = runningTime
        self.n = n
        self.kNN = kNN
        self.dataSize=dataSize


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

    def __init__(self, kNN, n, data, calculateDistance):
        self.kNN = kNN
        self.n = n
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
        self.edge_knn = [0]*size
        for i in range(0, size):
            distances = []
            for j in range(0, size):
                if i == j:
                    continue
                distances.append(self.distanceHandler.distance(i, j))
            distances.sort()
            self.edge_knn[i] = mean(distances[:self.kNN])

        self.db_outlier_indexes = list(range(0, size))
        self.__sortOutliersDesc()

        return Result(
            outlier_indexes=self.db_outlier_indexes[:self.n],
            outlier_scores=[self.edge_knn[x]
                            for x in self.db_outlier_indexes[:self.n]],
            n=self.n,
            kNN=self.kNN,
            calculations=len(self.distanceHandler.cache),
            runningTime=time()-startTime,
            dataSize=size
        )
