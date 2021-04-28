import outlier_detector.dhca
import numpy as np
from math import comb
import os


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

    kNN = 30
    n = 50
    kValues = [2, 3, 5, 8, 10, 15, 20, 50]

    fileName = 'out/corel_k_to_time.jsonl'
    count = 2
    while os.path.exists(fileName):
        fileName = 'out/corel_k_to_time_{}.jsonl'.format(count)
        count += 1
    print('Output File Path:', fileName)
    for k in kValues:
        print('Running for k:', k)
        runner = outlier_detector.dhca.Runner(
            kNN=kNN,
            k=k,
            n=n,
            maxClusterSize=50,
            data=data,
            calculateDistance=euclideanDistance
        )

        result = runner.run()
        with open(fileName, 'a+') as f:
            f.write(result.toJSON())
            f.write('\n')


if __name__ == '__main__':
    main()
