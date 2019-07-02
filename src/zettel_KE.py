import operator
import numpy as np
import zettel_preprocessor as process
import re

class KE(object):

    # def __init__(self):

    # def __init__(self, stop_words, regex='[\W\n]+'):
    #     #lets users call predefined stopwords easily in a platform agnostic manner or use their own list
    #     if isinstance(stop_words, list):
    #         self.__stop_words_pattern = build_stop_word_regex(stop_words)
    #     else:
    #         self.__stop_words_pattern = build_stop_word_regex(load_stop_words(stop_words, regex))

    # run -> tokenize, POS, lemmatize, bi-gram/tri-gram swaps, word scores, keyword scores, tf_idf, text rank, weight distribution...
    def run(self, zettels): #TODO fix self references
        z_process = process.ZettelPreProcessor()
        z_process.init_zettels(zettels)

        tokens = z_process.tokens
        pos_tokens = z_process.pos_tagged_tokens
        lemma_tokens = z_process.lemmatized_tokens
        bi_grams = z_process.bi_gram #TODO
        tri_grams = z_process.tri_gram #TODO

        doc_count_dict = self.create_doc_count_dictionary(lemma_tokens, z_process.create_unique_corpus())
        TF_IDF = self.tf_idf(lemma_tokens, doc_count_dict)

        word_scores = self.create_word_score(lemma_tokens) #TODO
        keyword_scores = self.create_keyword_score(lemma_tokens, word_scores, 1) #TODO

        print(tokens)
        print(pos_tokens)
        print(lemma_tokens)
        print(bi_grams)
        print(tri_grams)
        print(doc_count_dict)
        print(TF_IDF)
        print(word_scores)
        print(keyword_scores)
    # def run(self, text, minCharacters=1, maxWords=5, minFrequency=1):
    #     sentence_list = split_sentences(text)
    #
    #     phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern, minCharacters, maxWords)
    #
    #     word_scores = calculate_word_scores(phrase_list)
    #
    #     keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores, minFrequency)
    #
    #     sorted_keywords = sorted(keyword_candidates.items(), key=operator.itemgetter(1), reverse=True)
    #     return sorted_keywords


    # page rank = if zettel A has link to zettel B, can be represented as a direct edge from A to B
    # after creating graph, weights = (1-d) + d * ( dot( graph, page_ranks) )
    # initialize page ranks as 1
    def create_page_rank(self, graph):
        new_graph = np.asarray(self.normalize_graph(graph))
        page_ranks = np.full((len(new_graph)), 1)
        d = 0.85
        iter = 0
        for itme in new_graph:
            iter += 1
            page_ranks = (1-d) + d * np.dot(new_graph, page_ranks)
            print(iter)
            print(page_ranks)

    # normalize graph
    def normalize_graph(self, graph):
        col_index = 0
        for col in graph:
            row_index = 0
            sum = 0
            for item in col:
                sum = sum + graph[row_index][col_index]
                row_index += 1
            row_index = 0
            for item in col:
                graph[row_index][col_index] = graph[row_index][col_index] / sum
                row_index += 1
            col_index += 1
        return graph

    def tf_idf(self, lemma_tokens, doc_count_dict):
        """ tf_idf = tf * idf """
        total_docs = len(lemma_tokens)
        all_tf_idf = []
        row_length = 0.0
        for zettel in lemma_tokens:
            total_words = len(zettel)
            tf_idf = []
            count_dict = self.create_count_dictionary(zettel)
            for word in zettel:
                # tf = (count of given word for a given zettel) / (total number of words for given zettel)
                tf = count_dict[word] / total_words
                # idf = (total number of documents) / (number of documents containing word)
                idf = total_docs / doc_count_dict[word]
                tf_idf_value = tf * idf
                tf_idf.append(tf_idf_value)
            if row_length < len(tf_idf):
                row_length = len(tf_idf)
            all_tf_idf.append(tf_idf)
        for row in all_tf_idf:
            while len(row) < row_length:
                row.append(0.0)
        return all_tf_idf

    def create_count_dictionary(self, tokens):
        word_count_dict = {}
        for word in tokens:
            word_count_dict.setdefault(word, 0)
            word_count_dict[word] += 1
        return word_count_dict

    def create_doc_count_dictionary(self, lemma_tokens, unique_lemmas):
        doc_count_dict = {}
        for zettel in lemma_tokens:
            for word in unique_lemmas:
                for token in zettel:
                    if token == word:
                        doc_count_dict.setdefault(word, 0)
                        doc_count_dict[word] += 1
                        break
        return doc_count_dict

    # https://github.com/fabianvf/python-rake/blob/master/RAKE/RAKE.py
    # word score = deg(word) / freq(word) -- deg(word) = word_list_len - 1 for each word in the phrase + freq(word)
    # single word = 0 degree, bi-gram = 1 degree, tri-gram = 2 degree #TODO experiment with diff numbers
    def create_word_score(self, lemma_tokens):
        word_freq = {}
        word_deg = {}
        for zettel in lemma_tokens:
            for word in zettel:
                word_list = re.split(" ", word)
                word_list_deg = len(word_list) - 1
                for new_word in word_list:
                    word_freq.setdefault(new_word, 0)
                    word_freq[new_word] = word_freq[new_word] + 1
                    word_deg.setdefault(new_word, 0)
                    word_deg[new_word] = word_deg[new_word] + word_list_deg
        word_score = {}
        for word in word_freq:
            word_deg[word] = word_deg[word] + word_freq[word]
            word_score.setdefault(word, 0)
            word_score[word] = word_deg[word] / (word_freq[word] * 1.0)
        return word_score

    # https://github.com/fabianvf/python-rake/blob/master/RAKE/RAKE.py
    # keyword score = sum of each word score in phrase
    def create_keyword_score(self, lemma_tokens, word_score, min_freq):
        keywords_score = {}
        for zettel in lemma_tokens:
            for word in zettel:
                if zettel.count(word) >= min_freq:
                    keywords_score.setdefault(word, 0)
                    word_list = re.split(" ", word)
                    score = 0
                    for new_word in word_list:
                        score += word_score[new_word]
                    keywords_score[word] = score
        return keywords_score


ke = KE()
# g = [[0, 0, 0.5, 0],
#      [0, 0, 0.5, 1],
#      [1, 0.5, 0, 0],
#      [0, 0.5, 0, 0]]
# ke.create_page_rank(g)

rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"
baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"

z_process = process.ZettelPreProcessor()
zettels = z_process.get_zettels_from_directory(baseball)

ke.run(zettels)