import outlier_detector.dhca
import Levenshtein
from math import comb


def readData():
    '''
    Reads the gene sequences from the file
    and stores as a dictionary
    '''
    f = open('data/human_gene/human_gene.data', "r")
    data = {}
    label = ''
    genes = ''
    count = 0
    if f.mode == 'r':
        fl = f.readlines()
        for x in fl:
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


def levenshteinDistance(d1, d2):
    # (Levenshtein.distance isn't recognized for some reason) so we add this -
    # pylint: disable=no-member
    return Levenshtein.distance(d1, d2)


def main():
    mp = readData()
    labels = list(mp.keys())
    data = list(mp.values())

    fileName = 'out/human_gene.jsonl'
    kNN = 30
    n = 50
    k = 3

    print('Running dhca outlier detector:')
    runner = outlier_detector.dhca.Runner(
        kNN=kNN,
        k=k,
        n=n,
        maxClusterSize=3,
        data=data,
        calculateDistance=levenshteinDistance
    )

    result = runner.run()
    print('Outlier Labels:', [labels[x] for x in result.outlier_indexes])
    print(result.toJSON())
    with open(fileName, 'a+') as f:
        f.write(result.toJSON())
        f.write('\n')


if __name__ == '__main__':
    main()
