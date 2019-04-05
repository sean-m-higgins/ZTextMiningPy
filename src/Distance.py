from math import *


class Distance:

    def calculate_distances(self, matrix, n):
        i = 1
        distances = [0]
        for row in matrix:
            if i == len(matrix):
                break
            current_rows = [row, matrix[i]]
            distance = Distance.distance_calculator(self, current_rows, n)
            distances.append(distance)
            i += 1
        return distances

    def distance_calculator(self, rows, n):
        switch = {
            0: Distance.euclidean_distance(self, rows[0], rows[1]),
            1: Distance.manhattan_distance(self, rows[0], rows[1]),
            2: Distance.minkowsky_distance(self, rows[0], rows[1], 3),
            3: Distance.cosine_distance(self, rows[0], rows[1]),
            4: Distance.jaccard_similarity(self, rows[0], rows[1]),
        }
        return switch.get(n)

    def create_distance_matrix(self, distances):
        distance_matrix = []
        new_distances = distances
        length = len(distances)
        for distance in distances:
            distance_matrix.append(new_distances)
            new_distances = new_distances[length-1: length] + new_distances[0: length-1]
        return distance_matrix

    #from https://dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
    def euclidean_distance(self, row_one, row_two):
        """ distance = sqrt( sum( (differences between Ai and Bi)(squared) ) ) """
        return sqrt(sum((pow(a-b, 2)) for a, b in zip(row_one, row_two)))

    def manhattan_distance(self, row_one, row_two):
        """ distance = abs(Ax - Bx) + abs(Ay - By) """
        return sum(abs(a-b) for a, b in zip(row_one, row_two))

    def minkowsky_distance(self, row_one, row_two, p_value):
        """ distance = ( sum( abs(Ai - Bi)^p )^1/p ) """
        return self.nth_root(sum(pow(abs(a-b),p_value) for a, b in zip(row_one, row_two)), p_value)

    def nth_root(self, value, n_root):
        root_value = 1/float(n_root)
        return value ** root_value

    def cosine_distance(self, row_one, row_two):
        """ similarity = [A dot B] / [Magnitude(A) * Magnitude(B)] """
        numerator = sum(a*b for a, b in zip(row_one, row_two))
        denominator = self.square_rooted(row_one)*self.square_rooted(row_two)
        return numerator/float(denominator)

    def square_rooted(self, row):
        return sqrt(sum([a*a for a in row]))

    def jaccard_similarity(self, row_one, row_two):
        """ similarity = [A intersect B] / [A Union B] """
        intersection = len(list(set(row_one).intersection(row_two)))
        union = (len(row_one) + len(row_two)) - intersection
        return intersection/float(union)

    #TODO TF-IDF


