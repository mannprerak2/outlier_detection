import outlier_detector.simple_knn
from math import comb
import pandas as pd
import numpy as np

def readData():
    '''
    Reads the gene sequences from the file
    and stores as a dataframe.
    '''

    def weightConverter(x): return float(x)/2.2046226218  # Pounds to kilogram
    def heightConverter(x): return float(x)/39.3700787    # Inch to metre
    df = pd.read_csv('data/weight_height/weight_height.csv',
                     converters={
                         'Weight': weightConverter,
                         'Height': heightConverter
                     })
    # Remove gender information
    df.drop(['Gender'], axis = 1)

    return df


def distance(d1, d2):
    return np.linalg.norm(d1[1:]-d2[1:])


def main():
    df = readData()
    keys = list(df.keys())
    data = df.values

    fileName = 'out/weights_heights_simple.jsonl'
    kNN = 30
    n = 50

    print('Running simple outlier detector:')
    runner = outlier_detector.simple_knn.Runner(
        kNN=kNN,
        n=n,
        data=data,
        calculateDistance=distance
    )

    result = runner.run()
    with open(fileName, 'a+') as f:
            f.write(result.toJSON())
            f.write('\n')


if __name__ == '__main__':
    main()
