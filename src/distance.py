from src import zettel_preprocessor
from math import *
import numpy as np


class Distance:

    def calculate_distances(self, matrix, n):
        i = 1
        distances = [0.0]
        for row in matrix:
            if i == len(matrix):
                break
            current_rows = [row, matrix[i]]
            distance = float(self.distance_calculator(current_rows, n))
            distances.append(distance)
            i += 1
        return distances

    def distance_calculator(self, rows, n):
        if n == 0:
            return self.euclidean_distance(rows[0], rows[1])
        elif n == 1:
            return self.manhattan_distance(rows[0], rows[1])
        elif n == 2:
            return self.minkowsky_distance(rows[0], rows[1], 3)
        elif n == 3:
            return self.cosine_distance(rows[0], rows[1])
        elif n == 4:
            return self.jaccard_similarity(rows[0], rows[1])

    def get_distances(self, count_matrix, distance_str):
        if distance_str == 'euclidean':
            return self.calculate_distances(count_matrix, 0)
        elif distance_str == 'manhattan':
            return self.calculate_distances(count_matrix, 1)
        elif distance_str == 'minkowski':
            return self.calculate_distances(count_matrix, 2)
        elif distance_str == 'cosine':
            return self.calculate_distances(count_matrix, 3)
        elif distance_str == 'jaccard':
            return self.calculate_distances(count_matrix, 4)

    def create_distance_matrix(self, distances):
        distance_matrix = []
        distance_matrix.append(distances)
        new_distances = np.array(distances)
        for i in range(len(distances)-1):
            temp_dist = []
            temp_dist.append(new_distances[-1])
            new_distances = new_distances[0: -1]
            for int in new_distances:
                temp_dist.append(int)
            new_distances = temp_dist
            distance_matrix.append(new_distances)
        return distance_matrix

    def get_distance_matrix(self, count_matrix, dist_matrix_str):
        if dist_matrix_str == 'euclidean':
            return self.create_distance_matrix(self.get_distances(count_matrix, 'euclidean'))
        elif dist_matrix_str == 'manhattan':
            return self.create_distance_matrix(self.get_distances(count_matrix, 'manhattan'))
        elif dist_matrix_str == 'minkowski':
            return self.create_distance_matrix(self.get_distances(count_matrix, 'minkowski'))
        elif dist_matrix_str == 'cosine':
            return self.create_distance_matrix(self.get_distances(count_matrix, 'cosine'))
        elif dist_matrix_str == 'jaccard':
            return self.create_distance_matrix(self.get_distances(count_matrix, 'jaccard'))

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

    def tf_idf(self, zettels):
        """ tf_idf = tf * idf """
        process = zettel_preprocessor.ZettelPreProcessor()
        process.init_zettels(zettels)
        doc_count_dict = process.create_doc_count_dictionary()
        total_docs = len(zettels)
        tf_idf = []
        row_length = 0.0
        for zettel in zettels:
            new_tf_idf = []
            process.init_zettels(zettel)
            count_dict = process.create_count_dictionary()
            total_words = len(process.lemmatized_tokens)
            for word in process.lemmatized_tokens:
                # tf = (count of given word for a given zettel) / (total number of words for given zettel)
                tf = count_dict[word] / total_words
                # idf = (total number of documents) / (number of documents containing word)
                idf = total_docs / doc_count_dict[word]
                tf_idf_value = tf * idf
                new_tf_idf.append(tf_idf_value)
            if row_length < len(new_tf_idf):
                row_length = len(new_tf_idf)
            tf_idf.append(new_tf_idf)
        for row in tf_idf:
            while len(row) < row_length:
                row.append(0.0)
        return tf_idf


if __name__ == "__main__":
    process = zettel_preprocessor.ZettelPreProcessor()

    distance = Distance()
    distance_type = 'euclidean'
    matrix = np.array(distance.get_distance_matrix(process.unique_count_matrix, distance_type))
    tf_idf = distance.tf_idf(process.zettels)
