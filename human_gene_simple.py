import outlier_detector.simple_knn
from math import comb

# Reuse functions from human_gene.
from human_gene import readData, levenshteinDistance

def main():
    mp = readData()
    labels = list(mp.keys())
    data = list(mp.values())

    fileName = 'out/human_gene_simple.jsonl'
    kNN = 30
    n = 50

    print('Running simple outlier detector:')
    runner = outlier_detector.simple_knn.Runner(
        kNN=kNN,
        n=n,
        data=data,
        calculateDistance=levenshteinDistance
    )

    result = runner.run()

    print('Outlier Labels:', [labels[x] for x in result.outlier_indexes])
    print('Outlier Indexes:', result.outlier_indexes)
    print('Calculations:', result.calculations,
          'out of', comb(len(data),2))
    print('Running Time: {:.2f} seconds'.format(result.runningTime))

    result.all_scores = runner.edge_knn
    with open(fileName, 'a+') as f:
        f.write(result.toJSON())
        f.write('\n')
if __name__ == '__main__':
    main()
