'''
Implementing an efficient DB outlier detection technique using a divisive
hierarchical clustering algorithm.
'''

import random
import distance
from statistics import mean

class KNN:
    '''
    Stores the k nearest neighbours of a data item.
    '''
    def __init__(self, kNN):
        self.kNN = kNN
        self.__average = None
        self.distances = []

    def getMax(self):
        return self.distances[-1]

    def setMax(self, value):
        self.distances[-1] = value
        self.__average = None
        sorted(self.distances)

    def average(self):
        if self.__average is None:
            self.__average = mean(self.distances)
        return self.__average


class Node:
    '''
    A Node is a cluster.
    '''

    def __init__(self, samples = []):
        # Indices of all samples under this cluster.
        self.sampleNumbers = []
        self.sampleNumbers.extend(samples)

        # Randomly chosen cluster centers.
        self.centroidSeeds = []

    def addSample(self, sample):
        self.sampleNumbers.append(sample)

    def DHCA(self, dist_knn, edge_knn, kNN, nodeArray,
             currentNode, k, data, maxClusterSize, threshold,
             distance: distance.Distance) -> None:
        '''
        Input:
        - dist_knn: sorted distances of kNN for each data item.
        - edge_knn: average of distances of kNN for eacch data item.
        - kNN: number of kNNs for a data item.
        - nodeArray: Array of Nodes(clusters).
        - currentNode: current Node of the Node array.
        - k: number of clusters at each step.
        - data: input data size.
        - maxClusterSize: max size allowed for a cluster.
        - threshold: The value used to filter.

        Output: (Returns None)
        - Updated dist_knn and edge_knn (inplace).
        - New nodes generated (<=k) and pushed to nodeArray (inplace).
        '''

        # Randomly select k elements from sampleNumbers
        self.centroidSeeds = random.sample(population=self.sampleNumbers, k=k)

        # Generate k new nodes and map them with the index.
        nodeMap= {}
        for center in self.centroidSeeds:
            nodeMap[center] = Node([center])

        for i in self.sampleNumbers:
            if i in self.centroidSeeds:
                continue
            # Loop over nodes that are not center.

            # Get nearest center
            j = distance.closest(i, self.centroidSeeds)

            if dist_knn[i].getMax() > distance.distance(i, j):
                dist_knn[i].setMax(distance.distance(i, j))

            if dist_knn[i].average() > threshold:
                nodeMap[j].addSample(i)

        for k, v in nodeMap:
            if len(v.sampleNumbers) > maxClusterSize:
                nodeArray.append(v)


def main():
    pass


if __name__ == '__main__':
    main()
