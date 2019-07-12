import numpy as np
import zettel_preprocessor as process
import weights
import re


class KE:

    def __init__(self, zettels):
        z_process = process.ZettelPreProcessor()
        z_process.init_zettels(zettels)
        z_weights = weights.Weights()
        self.lemma_tokens = z_process.lemmatized_tokens
        self.filter_n_grams(z_process.bi_gram, z_weights.n_gram_min_freq, 2)
        self.filter_n_grams(z_process.tri_gram, z_weights.n_gram_min_freq, 3)
        self.filter_pos()
        self.doc_count_dict = self.create_doc_count_dictionary(z_process.create_unique_corpus())
        self.window_size = 4
        self.score_weights = z_weights.all_score_weights
        self.pos_score_switch = z_weights.pos_switch
        self.z_area_switch = z_weights.z_area_switch
        self.keyword_n = z_weights.keyword_n

    def run(self, min_freq):
        """ Calculate scores, Combine all scores into one, and Get top n keywords """
        self.tf_idf_scores = self.tf_idf()
        self.word_scores = self.create_word_score()
        self.keyword_scores = self.create_keyword_score(self.word_scores, min_freq)
        self.text_ranks = self.create_text_rank()
        self.pos_scores = {}
        self.z_area_scores = {}
        self.create_pos_and_area_score()
        self.all_scores = self.weight_distribution(self.score_weights)
        return self.get_keywords()

    def filter_n_grams(self, n_grams, min_freq, n):
        """ remove infrequent n_grams and add frequent n_grams to corr. zettel in lemma_tokens """
        all_n_grams = []
        for zettel in n_grams:
            cur_n_grams = []
            for gram in zettel:
                if zettel.count(gram) >= min_freq:
                    if gram not in cur_n_grams:
                        cur_n_grams.append(gram)
            all_n_grams.append(cur_n_grams)
        self.swap_n_grams(all_n_grams, n)

    def swap_n_grams(self, all_n_grams, n):  #TODO do this in preprocessing with tokens... or its fine
        all_new_tokens = self.lemma_tokens
        index = 0
        for zettel in all_n_grams:
            if len(zettel) != 0:
                for new_gram in zettel:
                    for token_zettel in all_new_tokens:
                        token_index = 0
                        for word in token_zettel:
                            if n == 2:
                                if token_index != len(token_zettel)-1:
                                    if word[0] + " " + token_zettel[token_index+1][0] == new_gram:
                                        word[0] = new_gram
                                        word[1] = 'NG'
                                        del token_zettel[token_index+1]
                            if n == 3:
                                if token_index != len(token_zettel)-1:
                                    if token_index != len(token_zettel)-2:
                                        if word[0] + " " + token_zettel[token_index+1][0] + " " + token_zettel[token_index+2][0] == new_gram:
                                            word[0] = new_gram
                                            word[1] = 'NG'
                                            del token_zettel[token_index+1]
                                            del token_zettel[token_index+2]
                            token_index += 1
                index += 1
        self.lemma_tokens = all_new_tokens

    def create_doc_count_dictionary(self, unique_tokens):
        """ {word: doc_count} """
        doc_count_dict = {}
        for zettel in self.lemma_tokens:
            for word in unique_tokens:
                for token in zettel:
                    if token[0] == word:
                        doc_count_dict.setdefault(word, 0)
                        doc_count_dict[word] += 1
                        break
        return doc_count_dict

    def create_count_dictionary(self, tokens):
        """ {word: count} """
        word_count_dict = {}
        for word in tokens:
            word_count_dict.setdefault(word[0], 0)
            word_count_dict[word[0]] += 1
        return word_count_dict

    def tf_idf(self):
        """ tf_idf = tf * idf """
        all_tf_idf = {}
        total_docs = len(self.lemma_tokens)
        for zettel in self.lemma_tokens:
            total_words = len(zettel)
            count_dict = self.create_count_dictionary(zettel)
            for word in zettel:
                # tf = (count of given word for a given zettel) / (total number of words for given zettel)
                tf = count_dict[word[0]] / total_words
                # idf = (total number of documents) / (number of documents containing word)
                idf = total_docs / self.doc_count_dict[word[0]]
                tf_idf_value = tf * idf
                all_tf_idf[word[0]] = tf_idf_value
        return all_tf_idf

    # https://github.com/fabianvf/python-rake/blob/master/RAKE/RAKE.py
    # single word = 0 degree, bi-gram = 1 degree, tri-gram = 2 degree
    def create_word_score(self):
        """ word_score = deg(word) / freq(word) ... deg(word) = phrase_len - 1 + freq(word) """
        word_freq = {}
        word_deg = {}
        for zettel in self.lemma_tokens:
            for word in zettel:
                word_list = re.split(" ", word[0])
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
    def create_keyword_score(self, word_scores, min_freq):
        """ keyword_score = sum of each word_score in phrase """
        keywords_score = {}
        for zettel in self.lemma_tokens:
            for word in zettel:
                if zettel.count(word) >= min_freq:
                    keywords_score.setdefault(word[0], 0)
                    word_list = re.split(" ", word[0])
                    score = 0
                    for new_word in word_list:
                        score += word_scores[new_word]
                    keywords_score[word[0]] = score
        return keywords_score

    # use all tokens, or optionally use only noun, proper noun, and verb tags TODO experiment...
    def filter_pos(self):
        """ remove words not of desired pos """
        all_tokens = []
        for zettel in self.lemma_tokens:
            tokens = []
            for word in zettel:
                if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'NG']:   # NG = n_gram
                    tokens.append(word)
            all_tokens.append(tokens)
        self.lemma_tokens = all_tokens

    # https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
    def create_text_rank(self):
        """ text rank = weight based on any two word pairs of words (undirected edge from 1 to 2)"""
        # filtered_tokens = self.filter_pos()    #if use, replace 2 self.lemma_tokens below
        vocab = self.create_vocab(self.lemma_tokens)
        token_windows = self.create_token_windows(self.lemma_tokens)
        graph = self.create_matrix(vocab, token_windows)
        text_rank = np.array([1] * len(vocab))
        previous_tr = 0
        d = 0.85
        min_difference = 1e-5
        for epoch in range(10):
            text_rank = (1 - d) + d * np.dot(graph, text_rank)
            if abs(previous_tr - sum(text_rank)) < min_difference:
                break
            else:
                previous_tr = sum(text_rank)
        node_weight = {}
        for word in vocab:
            node_weight[word] = text_rank[vocab[word]]
        return node_weight

    def create_vocab(self, filtered_tokens):
        """ {word: index} """
        vocab = {}
        index = 0
        for zettel in filtered_tokens:
            for word in zettel:
                if word[0] not in vocab:
                    vocab[word[0]] = index
                    index += 1
        return vocab

    # set window size k --> [w1, w2, …, w_k], [w2, w3, …, w_{k+1}], [w3, w4, …, w_{k+2}]...
    def create_token_windows(self, filtered_tokens):
        """ token pairs inside each window """
        all_token_pairs = []
        for zettel in filtered_tokens:
            for i, word in enumerate(zettel):
                for j in range(i+1, i+ self.window_size):
                    if j >= len(zettel):
                        break
                    pair = (word[0], zettel[j][0])
                    if pair not in all_token_pairs:
                        all_token_pairs.append(pair)
        return all_token_pairs

    def create_matrix(self, vocab, token_pairs):
        """ graph with undirected edges from first word to second word of pair """
        vocab_size = len(vocab)
        graph = np.zeros((vocab_size, vocab_size), dtype='float')
        for word_1, word_2 in token_pairs:
            i, j = vocab[word_1], vocab[word_2]
            graph[i][j] = 1
        graph = graph + graph.T - np.diag(graph.diagonal()) # symmetrize matrix
        norm = np.sum(graph, axis=0) # normalize matrix
        graph = np.divide(graph, norm, where= norm!= 0) #ignore the elements that = 0 in norm
        return graph

    def create_pos_and_area_score(self):
        """ pos_score = ('NN', .40) ('NNS', .35) ('NNP', .80) ('NNPS', .70) ('NG', .50) (V: .25) (Other: .15)
            z_area_score = (title: .80) (summary: .60) (note: 40) """
        for zettel in self.lemma_tokens:
            for word in zettel:
                self.pos_scores.setdefault(word[0], 0)
                self.pos_scores[word[0]] = self.pos_score_switch.get(word[1], 0)
                self.z_area_scores.setdefault(word[0], 0)
                self.z_area_scores[word[0]] = self.z_area_switch.get(word[2])

    def weight_distribution(self, weights):
        """ combine all scores together with weights """
        all_scores = []
        for zettel in self.lemma_tokens:
            scores = []
            for word in zettel:
                cur_tf_idf = self.tf_idf_scores[word[0]] / 3 #range: 0-3+
                if word[1] == 'NG':
                    word_list = re.split(" ", word[0])
                    cur_word_score = 0
                    i = 0
                    for new_word in word_list:
                        cur_word_score += self.word_scores[new_word]
                        i += 1
                    cur_word_score = cur_word_score / i / 2  #range: 0-2+
                else:
                    cur_word_score = self.word_scores[word[0]] / 2  #range: 0-2+
                cur_keyword_score = self.keyword_scores[word[0]] / 4  #0-4+
                cur_text_rank = self.text_ranks[word[0]] / 10  #range: 0-12+
                cur_pos_score = self.pos_scores[word[0]]
                cur_area_score = self.z_area_scores[word[0]]
                cur_total_score = ((cur_tf_idf * weights[0]) + (cur_word_score * weights[1]) +
                                   (cur_keyword_score * weights[2]) + (cur_text_rank * weights[3]) +
                                   (cur_pos_score * weights[4]) + (cur_area_score * weights[5])) / 6
                scores.append(cur_total_score)
            all_scores.append(scores)
        return all_scores

    def get_keywords(self):
        """ get top n keywords based on total score for each zettel """
        all_keywords = []
        z_index = 0
        for zettel in self.lemma_tokens:
            keywords = []
            w_index = 0
            cur_zettel_dict = {}
            for word in zettel:
                cur_zettel_dict.setdefault(word[0], 0)
                cur_word_total_score = self.all_scores[z_index][w_index]
                if cur_zettel_dict[word[0]] > cur_word_total_score:
                    w_index += 1
                    continue
                else:
                    cur_zettel_dict[word[0]] = cur_word_total_score
                    w_index += 1
            cur_sorted = sorted(cur_zettel_dict.items(), key=lambda kv: kv[1], reverse=True)
            for i in range(self.keyword_n):
                keywords.append(str(cur_sorted[i]))
            z_index += 1
            all_keywords.append(keywords)
        return all_keywords


# TODO delete? possibility of using page_rank by way of linking zettels together. if one zettel points to another could
#  mean those repitive words from those zettels are important keywords...?
# after creating graph, weights = (1-d) + d * ( dot( graph, page_ranks) )
# initialize page ranks as 1
# def create_page_rank(self, graph):
#     #     new_graph = np.asarray(self.normalize_graph(graph))
#     #     page_ranks = np.full((len(new_graph)), 1)
#     #     d = 0.85
#     #     iter = 0
#     #     for itme in new_graph:
#     #         iter += 1
#     #         page_ranks = (1-d) + d * np.dot(new_graph, page_ranks)
#     #         print(iter)
#     #         print(page_ranks)


rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"
baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
movies = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/movies"
clean_baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"

import datetime
print(datetime.datetime.now())

z_process = process.ZettelPreProcessor()
zettels = z_process.get_zettels_from_clean_directory(movies)

ke = KE(zettels)
suggested_keywords = ke.run(min_freq=1)

index = 0
for zettel in suggested_keywords:
    print("\nSuggested Keywords for Zettel " + str(index) + ": ")
    inner_i = 1
    for item in zettel:
        print(str(inner_i) + ": ")
        print(item)
        inner_i += 1
    index += 1

print("Done.")
print(datetime.datetime.now())
