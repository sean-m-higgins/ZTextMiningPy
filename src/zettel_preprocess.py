import numpy as np  # TODO
import nltk
from nltk.stem import WordNetLemmatizer
import re


class ZettelPreProcessor:

    def __init__(self, zet):
        self.lemmatizer = WordNetLemmatizer() # TODO keep here? slow?
        self.zettel = zet
        self.given_tags = []
        self.tokens = self.tokenizer()
        stopwords_file = open("../docs/data/processedData/stopWords/zettelStopWords.txt", "r")
        self.stop_words = [line[:-1] for line in stopwords_file.readlines()]  # removes /n from each line  # TODO remove title, note... from file
        self.filtered_words = self.remove_stop_words()
        self.pos_tagged_tokens = self.pos_tagger()
        self.lemmatized_tokens = self.create_lemmatized_tokens()
        self.bi_gram = self.create_n_gram(2)
        self.tri_gram = self.create_n_gram(3)
        self.filter_n_grams(n_grams=self.bi_gram, min_freq=2, n=2)
        self.filter_n_grams(n_grams=self.tri_gram, min_freq=2, n=3)
        self.filter_pos()  # TODO move in order

    def tokenizer(self):
        """ Split zettels by word """
        all_tokens = []  #TODO
        index = 0
        for section in self.zettel:
            if index == len(self.zettel) - 1:
                new_tags = re.split(";", section)
                self.given_tags.append(new_tags)  #TODO
            else:
                tokens = re.split('\W+', section)  #TODO
                tokens = list(filter(None, tokens))
                all_tokens.append(tokens)  #TODO
            index += 1
        return all_tokens

    def remove_stop_words(self):
        """ Remove any words included in stop words list """
        all_filtered_words = []  #TODO
        for section in self.tokens:
            new_section = []  #TODO
            for word in section:
                if word not in self.stop_words and word.lower() not in self.stop_words:
                    new_section.append(word)  #TODO
            all_filtered_words.append(new_section)  #TODO
        return all_filtered_words

    def pos_tagger(self):
        """ Tag each word with its part of speech """
        all_tokens_with_pos_tags = []  #TODO
        aread_id = 0
        for section in self.filtered_words:
            tags = nltk.pos_tag(section)
            for word in tags:
                if word[1].startswith('J'):
                    all_tokens_with_pos_tags.append([word[0], word[1], 'a', aread_id])  #TODO
                elif word[1].startswith('V'):
                    all_tokens_with_pos_tags.append([word[0], word[1], 'v', aread_id])  #TODO
                elif word[1].startswith('N'):
                    all_tokens_with_pos_tags.append([word[0], word[1], 'n', aread_id])  #TODO
                elif word[1].startswith('R'):
                    all_tokens_with_pos_tags.append([word[0], word[1], 'r', aread_id])  #TODO
            aread_id += 1
        return all_tokens_with_pos_tags

    def filter_pos(self):  #TODO
        """ remove words not of desired pos """
        all_tokens = []  #TODO
        for word in self.lemmatized_tokens:
            if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'NG']:  # NG = n_gram
                all_tokens.append(word)  #TODO
        self.lemmatized_tokens = all_tokens

    def create_lemmatized_tokens(self):
        """ Return lemmatized version for each word """
        all_lemmatized_tokens = []  #TODO
        for word in self.pos_tagged_tokens:
                all_lemmatized_tokens.append([self.lemmatizer.lemmatize(word[0], word[2]), word[1], word[3]])  #TODO
        return all_lemmatized_tokens

    def create_unique_corpus(self):
        """ Create distinct set of words """
        token_set = []  #TODO
        for word in self.lemmatized_tokens:
            if word[0] not in token_set:
                token_set.append(word[0])  #TODO
        return token_set

    def create_unique_tag_corpus(self):  #TODO
        """ Create distinct set of tags"""
        token_set = []  #TODO
        for word in self.given_tags:
            if word not in token_set:
                token_set.append(word)  #TODO
        return token_set

    def create_n_gram(self, n):
        """ Create list of candidate n-grams """
        all_n_grams = []  #TODO
        for section in self.tokens:
            for index in range(len(section) - n + 1):
                set = section[index:index + n]  #TODO
                if (set[0].lower() in self.stop_words) or (
                        set[n - 1].lower() in self.stop_words):  # skip if begin/end with stop_word
                    continue
                split_str = ""
                for i in range(n):
                    split_str += set[i]
                    if i < n - 1:
                        split_str += " "
                all_n_grams.append(split_str)  #TODO
        return all_n_grams

    def filter_n_grams(self, n_grams, min_freq, n):  #TODO min_freq
        """ remove infrequent n_grams and add frequent n_grams to respective zettel in lemma_tokens """
        all_n_grams = []  #TODO
        for gram in n_grams:
            if n_grams.count(gram) >= min_freq:  #TODO?
                if gram not in all_n_grams:
                    all_n_grams.append(gram)  #TODO
        self.swap_n_grams(all_n_grams, n)

    def swap_n_grams(self, all_n_grams, n):  # TODO
        """ swap list of candidate n_grams with their set of unigrams in the corpus """
        all_new_tokens = self.lemmatized_tokens
        zettel_size = len(all_new_tokens)  #TODO
        for new_gram in all_n_grams:
            token_index = 0
            for word in all_new_tokens:
                if n == 2:
                    if token_index != zettel_size - 1:
                        second_word = all_new_tokens[token_index + 1]  # TODO
                        if word[0] + " " + str(second_word) == new_gram:
                            word[0] = new_gram
                            word[1] = 'NG'
                            del all_new_tokens[token_index + 1]  #TODO
                            break
                if n == 3:
                    if token_index != zettel_size - 1:
                        if token_index != zettel_size - 2:
                            second_word = all_new_tokens[token_index + 1]  # TODO
                            third_word = all_new_tokens[token_index + 2]  # TODO
                            if word[0] + " " + str(second_word) + " " + str(third_word) == new_gram:
                                word[0] = new_gram
                                word[1] = 'NG'
                                del all_new_tokens[token_index + 1]  #TODO
                                del all_new_tokens[token_index + 2]  #TODO
                                break
                token_index += 1
        self.lemmatized_tokens = all_new_tokens
