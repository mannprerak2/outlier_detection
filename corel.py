import outlier_detector as algo
import numpy as np

def readData():
    '''
    Reads corel data and stores it as an array of shape(total, 9).
    '''
    f = open('data/corel/ColorMoments.asc', "r")
    data = {}
    count=0
    while True:
        count+=1
        # Get next line from file
        line = f.readline()
        if not line:
            break
        if count > 2000:
            break
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

    result.outlier_indexes.sort()
    print('Outlier Labels:', [labels[x] for x in result.outlier_indexes])
    print('Outlier Indexes:', result.outlier_indexes)
    print('Verified:', sum(result.verifiedStatus),
          'out of', len(result.verifiedStatus))
    print('Calculations:', result.calculations,
          'out of', (len(data)**2//2), 'i.e (N*N)/2')
    print('Running Time: {:.2f} seconds'.format(result.runningTime))


if __name__ == '__main__':
    main()
