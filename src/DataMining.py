from src import ZettelPreProcessor as PreProcess
from src import Distance

import os


class DataMining:
    baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
    bibs = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/bibs"
    examples = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/examples"
    rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"

    def get_zettels_from_directory(self, directory):
        zettels = []
        files = os.listdir(directory)
        for file in files:
            path = directory + "/" + file
            new_file = open(path, "r")
            contents = [new_file.read()]
            zettels.append(contents)
        return zettels

    zettels = get_zettels_from_directory("", baseball)

    process = PreProcess.ZettelPreProcessor()
    process.init_zettels(zettels)

    tokens = process.tokenizer()
    unique_corpus = process.create_unique_corpus(tokens)
    count_matrix = process.create_count_matrix(unique_corpus)
    n_gram = process.create_n_gram(tokens, 2)
    filtered_words = process.remove_stop_words(tokens)
    unique_tag_corpus = process.create_unique_tag_corpus(tokens)
    tag_boolean_matrix = process.create_boolean_tag_matrix(unique_tag_corpus)


    distance = Distance.Distance()

    euclidean = distance.calculate_distances(count_matrix, 0)
    manhattan = distance.calculate_distances(count_matrix, 1)
    minkowsky = distance.calculate_distances(count_matrix, 2)
    cosine = distance.calculate_distances(count_matrix, 3)
    jaccard = distance.calculate_distances(count_matrix, 4)

    euclidean_distance_matrix = distance.create_distance_matrix(euclidean)
    manhattan_distance_matrix = distance.create_distance_matrix(manhattan)
    minkowsky_distance_matrix = distance.create_distance_matrix(minkowsky)
    cosine_distance_matrix = distance.create_distance_matrix(cosine)
    jaccard_distance_matrix = distance.create_distance_matrix(jaccard)



#1 Convert source to List of String
#2 Make list of words unique
#3 Make 1&2 a function
#4 Create class to process all zettels
#5 Form unique word corpus
#TODO #6 Apply hierarchical clustering methods agglomerative, ... (spark)kmeans
#TODO Dendrogram?
#7 create matrix of word counts of the file's words in uniqueCorpus
#8 tokenize corpus
#TODO #9 stop words
#TODO #10 lemmatize/stem
#11 function to write to text file
#12 LDA
#13 function to create n-grams
#14 function to binarize?
#15 bag of words...(uniqueCorpus)
#16 test Suite
#17 distances  ... bootstrap? ... spearmans rank?... others? TODO TF-IDF
#18 part of speech??
#19 uniqueTagCorpus
#TODO #20 visual graphs
#TODO Tag/Autotag
