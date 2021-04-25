import algo
import Levenshtein


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


def levenshteinDistance(d1, d2):
    # (Levenshtein.distance isn't recognized for some reason) so we add this -
    # pylint: disable=no-member
    return Levenshtein.distance(d1, d2)


def main():
    mp = readHumanGeneData()
    labels = list(mp.keys())
    data = list(mp.values())

    runner = algo.Runner(
        kNN=3,
        k=3,
        n=3,
        maxClusterSize=3,
        data=data,
        calculateDistance=levenshteinDistance
    )

    result = runner.run()

    print('Outlier Labels:', [labels[x] for x in result.outlier_indexes])
    print('Outlier Indexes:', result.outlier_indexes)
    print('Verified:',sum(result.verifiedStatus),'out of',len(result.verifiedStatus))
    print('Calculations:',result.calculations,'out of',(len(data)**2//2), 'i.e (N*N)/2')
    print('Running Time: {:.2f} seconds'.format(result.runningTime))
if __name__ == '__main__':
    main()
