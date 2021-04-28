import outlier_detector
from math import comb

# Reuse functions from human_gene.
from human_gene import readData, levenshteinDistance

def main():
    mp = readData()
    labels = list(mp.keys())
    data = list(mp.values())

    print('Running simple outlier detector:')
    runner = outlier_detector.simple_knn.Runner(
        kNN=3,
        n=3,
        data=data,
        calculateDistance=levenshteinDistance
    )

    result = runner.run()

    print('Outlier Labels:', [labels[x] for x in result.outlier_indexes])
    print('Outlier Indexes:', result.outlier_indexes)
    print('Outlier Scores', [result.edge_knn[x] for x in result.outlier_indexes])
    print('Calculations:', result.calculations,
          'out of', comb(len(data),2))
    print('Running Time: {:.2f} seconds'.format(result.runningTime))
if __name__ == '__main__':
    main()
