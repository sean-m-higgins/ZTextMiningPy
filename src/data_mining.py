from src import zettel_preprocessor, distance, cluster
import pandas as pd
import numpy as np


class DataMining:
    baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
    bibs = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/bibs"
    examples = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/examples"
    rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"

    process = zettel_preprocessor.ZettelPreProcessor()
    zettels = process.get_zettels_from_directory(baseball)
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
    distance_type = 'manhattan'
    matrix = np.array(distance.get_distance_matrix(unique_count_matrix, distance_type))
    tf_idf = distance.tf_idf(zettels)

    cluster = cluster.Cluster()
    # hierarchical_cluster = cluster.hclust(matrix, distance_type)
    # hierarchical_cluster = cluster.hclust(tf_idf, 'tf idf')
    k_means = cluster.k_means(matrix, distance_type)


'#   #1 Convert source to List of String' \
    '#2 Make list of words unique' \
    '#3 Make 1&2 a function' \
    '#4 Create class to process all zettels' \
    '#5 Form unique word corpus' \
    '#6 Apply hierarchical clustering methods agglomerative, ... kmeans...??' \
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
    'TODO??#21 normalize tf_idf' \
    'TODO #22 retrieval class'
