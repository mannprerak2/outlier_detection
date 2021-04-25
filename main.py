'''
Implementing an efficient DB outlier detection technique using a divisive
hierarchical clustering algorithm.
'''

import random
import distance
from statistics import mean, stdev


class KNNValues:
    '''
    Stores the k nearest neighbours of a data item.
    '''

    def __init__(self, kNN, distances):
        self.kNN = kNN
        self.__average = None
        self.distances = distances
        self.distances.sort()

    def getMax(self):
        return self.distances[-1]

    def setMax(self, value):
        self.distances[-1] = value
        self.__average = None
        self.distances.sort()

    def average(self):
        if self.__average is None:
            self.__average = mean(self.distances)
        return self.__average


class Node:
    '''
    A Node is a cluster.
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
             distance: distance.Distance, verifiedStatus) -> None:
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


def readHumanGeneData():
    '''
    Reads the gene sequences from the file
    and stores as a dictionary
    '''
    f = open('data/human_gene.data', "r")
    data = {}
    label = ''
    genes = ''
    count = 0
    if f.mode == 'r':
        fl = f.readlines()
        for x in fl:
            # print(x)
            if x[0] == '>':
                count = count + 1
                if label != '':
                    data[label] = genes
                    genes = ''
                label = x[4:].rstrip()
            else:
                genes = genes + x.rstrip()
        data[label] = genes
    return data


def getThreshold(edge_knn):
    return mean(edge_knn)+stdev(edge_knn)


def areTopNVerified(n, db_outlier_indexes, verifiedStatus):
    for i in db_outlier_indexes[:n]:
        if not verifiedStatus[i]:
            return False
    return True


def getTopKUnverified(k, db_outlier_indexes, verifiedStatus):
    ans = []
    for i in db_outlier_indexes:
        if not verifiedStatus[i]:
            ans.append(i)
            if len(ans) >= k:
                break
    return ans


def main():
    mp = readHumanGeneData()

    labels = list(mp.keys())
    data = list(mp.values())
    size = len(labels)

    sampleIndexes = list(range(0, size))

    # Array of verification status of all data items. True if verified.
    verifiedStatus = [False]*size

    kNN = 3
    k = 3
    maxClusterSize = 4

    dist_knn = []
    edge_knn = [0]*size

    distanceFinder = distance.Distance(data)

    # Fill sequential dist_knn values for initial upper bound.
    for i in range(0, size-kNN):
        distances = []
        for j in range(0, kNN):
            distances.append(distanceFinder.distance(i, i+j+1))
        dist_knn.append(KNNValues(kNN, distances))

    for i in range(size-kNN, size):
        distances = []
        for j in range(0, kNN):
            distances.append(distanceFinder.distance(i-j-1, i))
        dist_knn.append(KNNValues(kNN, distances))

    # Set edge_knn (db outlier score).
    for i in range(0, len(edge_knn)):
        edge_knn[i] = dist_knn[i].average()

    # Reverse Sorted list of indexes with key=edge_knn value.
    db_outlier_indexes = list(range(0, size))
    db_outlier_indexes.sort(reverse=True, key=lambda x: edge_knn[x])

    # Number of outliers to find.
    n = 3

    while not areTopNVerified(n, db_outlier_indexes, verifiedStatus):
        topKUnverified = getTopKUnverified(
            k, db_outlier_indexes, verifiedStatus)

        nodeArray = [Node(sampleIndexes=sampleIndexes,
                          centroidSeeds=topKUnverified, verifyCenters=True)]

        threshold = getThreshold(edge_knn)

        for node in nodeArray:
            node.DHCA(dist_knn=dist_knn, edge_knn=edge_knn,
                      kNN=kNN, nodeArray=nodeArray, k=k, maxClusterSize=maxClusterSize,
                      distance=distanceFinder, threshold=threshold, verifiedStatus=verifiedStatus)

        # Sort db outliers
        db_outlier_indexes.sort(reverse=True, key=lambda x: edge_knn[x])
        print(db_outlier_indexes[:n])


if __name__ == '__main__':
    main()
