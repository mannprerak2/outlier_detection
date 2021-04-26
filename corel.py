import dhca_outlier_detector as algo
import numpy as np
from math import comb


def readData():
    '''
    Reads corel data and stores it as an array of shape(total, 9).
    '''
    f = open('data/corel/ColorMoments.asc', "r")
    data = {}
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = f.readline()
        if not line:
            break
        # if count > 300:
        #     break
        line = line.split(' ')
        data[line[0]] = np.array(list(map(float, line[1:])))
    f.close()
    return data


def euclideanDistance(d1, d2):
    return np.linalg.norm(d1-d2)


def main():
    mp = readData()
    labels = list(mp.keys())
    data = list(mp.values())

    runner = algo.Runner(
        kNN=20,
        k=5,
        n=10,
        maxClusterSize=50,
        data=data,
        calculateDistance=euclideanDistance
    )

    result = runner.run()

    print('Outlier Labels:', [labels[x] for x in result.outlier_indexes])
    print(result.toJSON())


if __name__ == '__main__':
    main()
