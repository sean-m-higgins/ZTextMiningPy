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
    def run(self, zettels):
        z_process = process.ZettelPreProcessor()
        z_process.init_zettels(zettels)

        # tokens = z_process.tokens
        # pos_tokens = z_process.pos_tagged_tokens
        self.lemma_tokens = z_process.lemmatized_tokens

        print(z_process.bi_gram)
        print(z_process.tri_gram)

        self.filter_n_grams(z_process.bi_gram, 2)
        self.filter_n_grams(z_process.tri_gram, 2)


        doc_count_dict = self.create_doc_count_dictionary(z_process.create_unique_corpus())
        TF_IDF = self.tf_idf(doc_count_dict)

        self.word_scores = self.create_word_score() #TODO
        keyword_scores = self.create_keyword_score(1) #TODO

        self.window_size = 4
        text_ranks = self.create_text_rank()


        # print(tokens)
        # print(pos_tokens)
        # print(lemma_tokens)
        # print(bi_grams)
        # print(tri_grams)
        # print(doc_count_dict)
        # print(TF_IDF)
        # print(word_scores)
        # print(keyword_scores)
        # print(text_ranks)

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

    def tf_idf(self, doc_count_dict):
        """ tf_idf = tf * idf """
        total_docs = len(self.lemma_tokens)
        all_tf_idf = []
        row_length = 0.0
        for zettel in self.lemma_tokens:
            total_words = len(zettel)
            tf_idf = []
            count_dict = self.create_count_dictionary(zettel)
            for word in zettel:
                # tf = (count of given word for a given zettel) / (total number of words for given zettel)
                tf = count_dict[word[0]] / total_words
                # idf = (total number of documents) / (number of documents containing word)
                idf = total_docs / doc_count_dict[word[0]]
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
            word_count_dict.setdefault(word[0], 0)
            word_count_dict[word[0]] += 1
        return word_count_dict

    def create_doc_count_dictionary(self, unique_lemmas):
        doc_count_dict = {}
        for zettel in self.lemma_tokens:
            for word in unique_lemmas:
                for token in zettel:
                    if token[0] == word:
                        doc_count_dict.setdefault(word, 0)
                        doc_count_dict[word] += 1
                        break
        return doc_count_dict

    # https://github.com/fabianvf/python-rake/blob/master/RAKE/RAKE.py
    # word score = deg(word) / freq(word) -- deg(word) = word_list_len - 1 for each word in the phrase + freq(word)
    # single word = 0 degree, bi-gram = 1 degree, tri-gram = 2 degree #TODO experiment with diff numbers
    def create_word_score(self):
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
    # keyword score = sum of each word score in phrase
    def create_keyword_score(self, min_freq):
        keywords_score = {}
        for zettel in self.lemma_tokens:
            for word in zettel:
                if zettel.count(word) >= min_freq:
                    keywords_score.setdefault(word[0], 0)
                    word_list = re.split(" ", word[0])
                    score = 0
                    for new_word in word_list:
                        score += self.word_scores[new_word]
                    keywords_score[word[0]] = score
        return keywords_score

    # https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
    # text rank = calculate weight for each word based on graph. Any two word pairs have an undirected edge from 1 to 2
    def create_text_rank(self):
        filtered_tokens = self.filter_pos()
        vocab = self.create_vocab(filtered_tokens)
        token_windows = self.create_token_windows(filtered_tokens)
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

    # use all tokens, or optionally use only noun, proper noun, and verb tags TODO experiment...
    def filter_pos(self):
        all_tokens = []
        for zettel in self.lemma_tokens:
            tokens = []
            for word in zettel:
                if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'NG']: # NG = n_gram
                    tokens.append(word[0])
            all_tokens.append(tokens)
        return all_tokens

    # create (word, index) vocabulary
    def create_vocab(self, filtered_tokens):
        vocab = {}
        index = 0
        for zettel in filtered_tokens:
            for word in zettel:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
        return vocab

    # set window size k --> [w1, w2, …, w_k], [w2, w3, …, w_{k+1}], [w3, w4, …, w_{k+2}]...
    def create_token_windows(self, filtered_tokens):
        all_token_pairs = []
        for zettel in filtered_tokens:
            for i, word in enumerate(zettel):
                for j in range(i+1, i+ self.window_size):
                    if j >= len(zettel):
                        break
                    pair = (word, zettel[j])
                    if pair not in all_token_pairs:
                        all_token_pairs.append(pair)
        return all_token_pairs

    def create_matrix(self, vocab, token_windows):
        vocab_size = len(vocab)
        graph = np.zeros((vocab_size, vocab_size), dtype='float')
        for word_1, word_2 in token_windows:
            i, j = vocab[word_1], vocab[word_2]
            graph[i][j] = 1
        graph = graph + graph.T - np.diag(graph.diagonal()) # symmetrize matrix
        norm = np.sum(graph, axis=0) # normalize matrix
        graph = np.divide(graph, norm, where= norm!= 0) #ignore the elements that = 0 in norm
        return graph

    def filter_n_grams(self, n_grams, min_freq):
        all_n_grams = []
        for zettel in n_grams:
            cur_n_grams = []
            for gram in zettel:
                if zettel.count(gram) >= min_freq:
                    cur_n_grams.append(gram)
            all_n_grams.append(cur_n_grams)

        all_new_tokens = self.lemma_tokens
        print(all_new_tokens)
        index = 0
        for zettel in n_grams:
            for gram in zettel:
                all_new_tokens[index].append([gram, 'NG'])
            index += 1
        print(all_new_tokens)
        self.lemma_tokens = all_new_tokens


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


ke = KE()

rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"
baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"

z_process = process.ZettelPreProcessor()
zettels = z_process.get_zettels_from_directory(baseball)

ke.run(zettels)