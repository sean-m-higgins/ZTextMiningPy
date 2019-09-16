import numpy as np  # TODO
import nltk
from nltk.stem import WordNetLemmatizer
import re


class ZettelPreProcessor:

    def __init__(self, zets):
        self.lemmatizer = WordNetLemmatizer()
        self.zettels = zets
        self.given_tags = []
        self.tokens = self.tokenizer()
        stopwords_file = open("../docs/data/processedData/stopWords/zettelStopWords.txt", "r")
        self.stop_words = [line[:-1] for line in stopwords_file.readlines()]  # removes /n from each line  # TODO remove title, note... from file
        self.filtered_words = self.remove_stop_words()
        self.pos_tagged_tokens = self.pos_tagger()
        self.lemmatized_tokens = self.create_lemmatized_tokens()
        self.bi_gram = self.create_n_gram(2)
        self.tri_gram = self.create_n_gram(3)
        z_weights = Weights()
        self.filter_n_grams(self.bi_gram, z_weights.n_gram_min_freq, 2)
        self.filter_n_grams(self.tri_gram, z_weights.n_gram_min_freq, 3)
        self.filter_pos()
        self.doc_count_dict = self.create_doc_count_dictionary(self.create_unique_corpus())
        self.score_weights = z_weights.all_score_weights
        self.pos_score_switch = z_weights.pos_switch
        self.z_area_switch = z_weights.z_area_switch
        self.keyword_n = z_weights.keyword_n
        self.min_keyword_freq = z_weights.min_keyword_freq

    def tokenizer(self):
        """ Split zettels by word """
        all_tokens = []
        for zettel in self.zettels:
            new_zettel = []
            index = 0
            for section in zettel:
                if index == len(zettel) - 1:
                    new_tags = re.split(";", section)
                    self.given_tags.append(new_tags)
                else:
                    tokens = re.split('\W+', section)
                    tokens = list(filter(None, tokens))
                    new_zettel.append(tokens)
                index += 1
            all_tokens.append(new_zettel)
        return all_tokens

    def remove_stop_words(self):
        """ Remove any words included in stop words list """
        all_filtered_words = []
        for zettel in self.tokens:
            filtered_zettels = []
            for section in zettel:
                new_section = []
                for word in section:
                    if word not in self.stop_words and word.lower() not in self.stop_words:
                        new_section.append(word)
                filtered_zettels.append(new_section)
            all_filtered_words.append(filtered_zettels)
        return all_filtered_words

    def pos_tagger(self):
        """ Tag each word with its part of speech """
        all_tokens_with_pos_tags = []
        for zettel in self.filtered_words:
            tokens_with_pos_tags = []
            aread_id = 0
            for section in zettel:
                tags = nltk.pos_tag(section)
                for word in tags:
                    if word[1].startswith('J'):
                        tokens_with_pos_tags.append([word[0], word[1], 'a', aread_id])
                    elif word[1].startswith('V'):
                        tokens_with_pos_tags.append([word[0], word[1], 'v', aread_id])
                    elif word[1].startswith('N'):
                        tokens_with_pos_tags.append([word[0], word[1], 'n', aread_id])
                    elif word[1].startswith('R'):
                        tokens_with_pos_tags.append([word[0], word[1], 'r', aread_id])
                aread_id += 1
            all_tokens_with_pos_tags.append(tokens_with_pos_tags)
        return all_tokens_with_pos_tags

    def create_lemmatized_tokens(self):
        """ Return lemmatized version for each word """
        all_lemmatized_tokens = []
        for zettel in self.pos_tagged_tokens:
            lemmatized_tokens = []
            for word in zettel:
                lemmatized_tokens.append([self.lemmatizer.lemmatize(word[0], word[2]), word[1], word[3]])
            all_lemmatized_tokens.append(lemmatized_tokens)
        return all_lemmatized_tokens

    def create_n_gram(self, n):
        """ Create list of candidate n-grams """
        all_n_grams = []
        for zettel in self.tokens:
            n_grams = []
            for section in zettel:
                for index in range(len(section) - n + 1):
                    set = section[index:index + n]
                    if (set[0].lower() in self.stop_words) or (
                            set[n - 1].lower() in self.stop_words):  # skip if begin/end with stop_word
                        continue
                    split = ""
                    for i in range(n):
                        split += set[i]
                        if i < n - 1:
                            split = split + " "
                    n_grams.append(split)
            all_n_grams.append(n_grams)
        return all_n_grams

    def create_unique_corpus(self):
        """ Create distinct set of words """
        token_set = []
        for zettel in self.lemmatized_tokens:
            for word in zettel:
                if word[0] not in token_set:
                    token_set.append(word[0])
        return token_set

    def create_unique_tag_corpus(self):
        """ Create distinct set of tags"""
        token_set = []
        for zettel in self.given_tags:
            for word in zettel:
                if word not in token_set:
                    token_set.append(word)
        return token_set

    def filter_n_grams(self, n_grams, min_freq, n):
        """ remove infrequent n_grams and add frequent n_grams to respective zettel in lemma_tokens """
        all_n_grams = []
        for zettel in n_grams:
            cur_n_grams = []
            for gram in zettel:
                if zettel.count(gram) >= min_freq:
                    if gram not in cur_n_grams:
                        cur_n_grams.append(gram)
            all_n_grams.append(cur_n_grams)
        self.swap_n_grams(all_n_grams, n)

    def swap_n_grams(self, all_n_grams, n):  # TODO
        """ swap list of candidate n_grams with their set of unigrams in the corpus """
        all_new_tokens = self.lemmatized_tokens
        index = 0
        for zettel in all_n_grams:
            if len(zettel) != 0:
                for new_gram in zettel:
                    for token_zettel in all_new_tokens:
                        token_index = 0
                        for word in token_zettel:
                            if n == 2:
                                if token_index != len(token_zettel) - 1:
                                    if word[0] + " " + token_zettel[token_index + 1][0] == new_gram:
                                        word[0] = new_gram
                                        word[1] = 'NG'
                                        del token_zettel[token_index + 1]
                            if n == 3:
                                if token_index != len(token_zettel) - 1:
                                    if token_index != len(token_zettel) - 2:
                                        if word[0] + " " + token_zettel[token_index + 1][0] + " " + \
                                                token_zettel[token_index + 2][0] == new_gram:
                                            word[0] = new_gram
                                            word[1] = 'NG'
                                            del token_zettel[token_index + 1]
                                            del token_zettel[token_index + 2]
                            token_index += 1
                index += 1
        self.lemmatized_tokens = all_new_tokens

    def filter_pos(self):
        """ remove words not of desired pos """
        all_tokens = []
        for zettel in self.lemmatized_tokens:
            tokens = []
            for word in zettel:
                if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'NG']:  # NG = n_gram
                    tokens.append(word)
            all_tokens.append(tokens)
        self.lemmatized_tokens = all_tokens

    def create_doc_count_dictionary(self, unique_tokens):
        """ {word: doc_count} """
        doc_count_dict = {}
        for zettel in self.lemmatized_tokens:
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


class Weights:

    def __init__(self):
        # tf_idf @ 0; keyword_score @ 1; pos_score @ 2; area_score @ 3
        self.all_score_weights = [0.50, 0.30, 0.1, 0.1]
        self.pos_switch = {
            'NN': 0.30,
            'NNS': 0.25,
            'NNP': 0.80,
            'NNPS': 0.70,
            'NG': 0.70,
        }
        self.z_area_switch = {
            0: 0.80,
            1: 0.60,
            2: 0.40
        }
        self.n_gram_min_freq = 2
        self.keyword_n = 7
        self.min_keyword_freq = 1
