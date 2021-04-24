'''
Utility to provde distance between points.
'''
import Levenshtein

class Distance:
    def __init__(self, data):
        self.data = data
    def distance(self, i, j):
        # TODO: add caching.

        # (Levenshtein.distance isn't recognized for some reason) so we add this -
        # pylint: disable=no-member
        return Levenshtein.distance(self.data[i], self.data[j])

    def closest(self, i, arr) -> int:
        distances = [self.distance(i,x) for x in arr]
        return arr[distances.index(min(distances))]

