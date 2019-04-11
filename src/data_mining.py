from src import zettel_preprocessor, distance, cluster
import numpy as np
import os
import pandas as pd


class DataMining:
    baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
    bibs = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/bibs"
    examples = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/examples"
    rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"

    def get_zettels_from_directory(directory):
        global zettels
        zettels = []
        files = os.listdir(directory)
        for file in files:
            path = directory + "/" + file
            alist = [line.rstrip() for line in open(path)]
            contents = [alist]
            zettels.append(contents)
        return zettels

    get_zettels_from_directory(rheingold)

    process = zettel_preprocessor.ZettelPreProcessor()
    process.init_zettels(zettels)

    lemmatized_tokens = process.process_zettels()
    n_grams = process.create_n_gram(lemmatized_tokens, 2)

    unique_corpus = process.create_unique_corpus(lemmatized_tokens)
    unique_n_gram_corpus = process.create_unique_corpus(n_grams)
    #unique_tag_corpus = process.create_unique_tag_corpus(tags)  #TODO get tokens from a array and parse...

    #tag_boolean_matrix = process.create_boolean_tag_matrix(unique_tag_corpus)

    unique_count_matrix = process.create_count_matrix(unique_corpus)
    n_gram_count_matrix = process.create_count_matrix(unique_n_gram_corpus)
    #tag_count_matrix = process.create_unique_corpus(unique_tag_corpus)

    count_dictionary = process.create_count_dictionary(lemmatized_tokens)
    doc_count_dictionary = process.create_doc_count_dictionary(lemmatized_tokens)

    distance = distance.Distance()

    def calc_dist(self, distance_str):
        switch = {
            'euclidean': self.distance.calculate_distances(self.unique_count_matrix, 0),
            'manhattan': self.distance.calculate_distances(self.unique_count_matrix, 1),
            'minkowsky': self.distance.calculate_distances(self.unique_count_matrix, 2),
            'cosine': self.distance.calculate_distances(self.unique_count_matrix, 3),
            'jaccard': self.distance.calculate_distances(self.unique_count_matrix, 4),
            'tf_idf': self.distance.tf_idf(zettels),
        }
        return switch.get(distance_str)

    def calc_dist_matrix(self, dist_matrix_str):
        switch = {
            'euclidean': self.distance.create_distance_matrix(self.calc_dist('euclidean')),
            'manhattan': self.distance.create_distance_matrix(self.calc_dist('manhattan')),
            'minkowsky': self.distance.create_distance_matrix(self.calc_dist('minkowsky')),
            'cosine': self.distance.create_distance_matrix(self.calc_dist('cosine')),
            'jaccard': self.distance.create_distance_matrix(self.calc_dist('jaccard')),
        }
        return switch.get(dist_matrix_str)

    cluster = cluster.Cluster()

    matrix = np.array(calc_dist_matrix('manhattan'))
    hierarchical_cluser = cluster.hclust(matrix)


'#   #1 Convert source to List of String' \
    '#2 Make list of words unique' \
    '#3 Make 1&2 a function' \
    '#4 Create class to process all zettels' \
    '#5 Form unique word corpus' \
    'TODO #6 Apply hierarchical clustering methods agglomerative, ... kmeans' \
    '#7 create matrix of word counts of the files words in uniqueCorpus' \
    '#8 tokenize corpus' \
    '#9 remove stop words' \
    '#10 lemmatize/stem' \
    '#11 function to write to text file' \
    'TODO #12 LDA' \
    '#13 n-grams' \
    '#14 word,count dictionary' \
    '#15 visual graphs' \
    '#16 test Suite' \
    '#17 distances - euclidean, manhattan, cosine, jaccard, minkowski, tf_idf...bootstrap?...spearmans_rank?...??' \
    '#18 part of speech' \
    '#19 uniqueTagCorpus' \
    'TODO #20 Tag/Autotag' \
    'TODO #21 normalize tf_idf'
