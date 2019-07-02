import operator
import numpy as np

class KE(object):

    # def __init__(self, stop_words, regex='[\W\n]+'):
    #     #lets users call predefined stopwords easily in a platform agnostic manner or use their own list
    #     if isinstance(stop_words, list):
    #         self.__stop_words_pattern = build_stop_word_regex(stop_words)
    #     else:
    #         self.__stop_words_pattern = build_stop_word_regex(load_stop_words(stop_words, regex))
    #
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

    # run -> tokenize, POS, bi-gram/tri-gram, stopwords, stem/lemma, word scores, keyword scores, tf_idf
    # word score = deg(word) / freq(word) -- deg(word) = word_list_len - 1 for each word in the phrase + freq(word)
    # keyword score = sum of each word score in phrase

    # page rank = if zettel A has link to zettel B, can be represented as a direct edge from A to B
    # after creating graph, weights = (1-d) + d * ( dot( graph, page_ranks) )
    # initialize page ranks as 1
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

    def create_page_rank(self, graph):
        new_graph = np.asarray(self.normalize_graph(graph))
        page_ranks = np.full((len(new_graph)), 1)
        d = 0.85
        iter = 0

        for row in g:
            for item in row:
                iter += 1
                page_ranks = (1-d) + d * np.dot(new_graph, page_ranks)
                print(iter)
                print(page_ranks)


ke = KE()
g = [[0, 0, 0.5, 0],
     [0, 0, 0.5, 1],
     [1, 0.5, 0, 0],
     [0, 0.5, 0, 0]]
ke.create_page_rank(g)
