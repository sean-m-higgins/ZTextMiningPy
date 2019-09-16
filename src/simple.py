import numpy as np  # TODO
import nltk
import os
import re


class ZettelPreProcessor:  # TODO tags...

    def __init__(self, zets):
        self.zettels = zets
        self.given_tags = []
        self.tokens = self.tokenizer()
        stopwords_file = open("../docs/data/processedData/stopWords/zettelStopWords.txt", "r")
        self.stop_words = [line[:-1] for line in stopwords_file.readlines()]  # removes /n from each line  # TODO remove title, note... from file
        self.filtered_words = self.remove_stop_words()
        self.bi_gram = self.create_n_gram(2)
        self.tri_gram = self.create_n_gram(3)
        z_weights = Weights()
        self.doc_count_dict = self.create_doc_count_dictionary(self.create_unique_corpus())
        self.score_weights = z_weights.all_score_weights
        self.z_area_switch = z_weights.z_area_switch
        self.keyword_n = z_weights.keyword_n
        self.min_keyword_freq = z_weights.min_keyword_freq
        self.tf_idf_scores = self.tf_idf()
        self.z_area_scores = {}
        self.create_pos_and_area_score()
        self.all_scores = self.weight_distribution()
        self.all_scores_dict = self.get_all_scores_dict()

    def run(self):
        """ Get top n keywords """
        return self.get_keywords()

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
            area_id = 0
            for section in zettel:
                for word in section:
                    if word not in self.stop_words and word.lower() not in self.stop_words:
                        filtered_zettels.append([word, area_id])
                area_id += 1
            all_filtered_words.append(filtered_zettels)
        return all_filtered_words

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
        for zettel in self.filtered_words:
            for word in zettel:
                if word[0] not in token_set:
                    token_set.append(word[0])
        return token_set

    def create_unique_tag_corpus(self):
        """ Create distinct set of tags """
        token_set = []
        for zettel in self.given_tags:
            for word in zettel:
                if word not in token_set:
                    token_set.append(word)
        return token_set

    def create_doc_count_dictionary(self, unique_tokens):
        """ {word: doc_count} """
        doc_count_dict = {}
        for zettel in self.filtered_words:
            for word in zettel:
                for token in unique_tokens:
                    if token == word[0]:
                        doc_count_dict.setdefault(word[0], 0)
                        doc_count_dict[word[0]] += 1
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
        total_docs = len(self.filtered_words)
        for zettel in self.filtered_words:
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

    def create_pos_and_area_score(self):
        """ z_area_score = (title: .80) (summary: .60) (note: 40) """
        for zettel in self.filtered_words:
            for word in zettel:
                self.z_area_scores.setdefault(word[0], 0)
                self.z_area_scores[word[0]] = self.z_area_switch.get(word[1])

    def weight_distribution(self):
        """ combine all scores together with weights """
        all_scores = []
        for zettel in self.filtered_words:
            scores = []
            for word in zettel:
                cur_tf_idf = self.tf_idf_scores[word[0]] / 3  # range: 0-3+
                cur_area_score = self.z_area_scores[word[0]]
                cur_total_score = ((cur_tf_idf * self.score_weights[0]) +
                                   (cur_area_score * self.score_weights[2])) / 2
                scores.append(cur_total_score)
            all_scores.append(scores)
        return all_scores

    def get_all_scores_dict(self):
        z_index = 0
        new_all_scores = {}
        for zettel in self.filtered_words:
            w_index = 0
            for word in zettel:
                new_all_scores.setdefault(word[0], 0)
                new_all_scores[word[0]] = self.all_scores[z_index][w_index]
                w_index += 1
            z_index += 1
        return new_all_scores

    def split_check(self, split_word, split_keyword):
        """ if any of the words in split_word match any of the words in split_keyword, return true, else false """
        check = False
        for item in split_word:
            for key_item in split_keyword:
                if item.lower() == key_item.lower():
                    check = True
        return check

    def get_final_keywords(self, keywords_dict):
        """ Filter out any keywords that appear more than once, choosing the higher scored duplicate """
        new_keywords_dict = {}
        word_index = 0
        black_list = []
        for word in keywords_dict:
            keyword_index = 0
            for keyword in keywords_dict:
                if word_index != keyword_index:
                    keyword_index += 1
                    # if two words match, take higher scored phrase
                    if word.lower() == keyword.lower():
                        if keywords_dict[word] > keywords_dict[keyword] or keywords_dict[word] == keywords_dict[keyword]:
                            if keyword not in black_list:
                                black_list.append(keyword)
                            if word not in new_keywords_dict:
                                new_keywords_dict[word] = keywords_dict[word]
                                break
                            break
                        else:
                            if word not in black_list:
                                black_list.append(word)
                            if keyword not in new_keywords_dict:
                                new_keywords_dict[keyword] = keywords_dict[keyword]
                                break
                            break
                    elif self.split_check(re.split(" ", word), re.split(" ", keyword)):
                        if keywords_dict[word] > keywords_dict[keyword]:
                            if keyword not in black_list:
                                black_list.append(keyword)
                            if word not in new_keywords_dict:
                                new_keywords_dict[word] = keywords_dict[word]
                                break
                            break
                        else:
                            if word not in black_list:
                                black_list.append(word)
                            if keyword not in new_keywords_dict:
                                new_keywords_dict[keyword] = keywords_dict[keyword]
                                break
                            break
                else:
                    keyword_index += 1
            if word not in black_list:
                if word not in new_keywords_dict:
                    new_keywords_dict[word] = keywords_dict[word]
            word_index += 1
        return new_keywords_dict

    def get_keywords(self):
        """ get top n keywords based on total score for each zettel """
        all_keywords = []
        z_index = 0
        for zettel in self.filtered_words:
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
            final_zettel_dict = self.get_final_keywords(cur_zettel_dict)
            cur_sorted = sorted(final_zettel_dict.items(), key=lambda kv: kv[1], reverse=True)
            for i in range(self.keyword_n):
                keywords.append(str(cur_sorted[i]))
            z_index += 1
            all_keywords.append(keywords)
        return all_keywords


class Weights:

    def __init__(self):
        # tf_idf @ 0; keyword_score @ 1; pos_score @ 2; area_score @ 3
        self.all_score_weights = [0.70, 0.15, 0.15]
        self.z_area_switch = {
            0: 0.80,
            1: 0.60,
            2: 0.40
        }
        self.n_gram_min_freq = 2
        self.keyword_n = 7
        self.min_keyword_freq = 1


def get_zettels_from_directory(directory):
    new_zettels = []
    files = os.listdir(directory)
    for file in files:
        path = directory + "/" + file
        contents = [str([line.rstrip() for line in open(path)])]
        new_zettels.append(contents)
    return new_zettels


def get_zettels_from_clean_directory(directory):
    new_zettels = []
    files = os.listdir(directory)
    for file in files:
        path = directory + "/" + file
        zettel = []
        lines = open(path).readlines()
        for line in lines:
            zettel.append(line)
        new_zettels.append(zettel)
    return new_zettels


if __name__ == "__main__":
    baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
    bibs = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/bibs"
    examples = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/examples"
    rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"
    movies = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/movies"
    clean_baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"

    import datetime
    print(datetime.datetime.now())

    # zettels = process.get_zettels_from_directory(baseball)
    zettels = get_zettels_from_clean_directory(movies)
    # zettels = get_zettels_from_clean_directory(clean_baseball)  # TODO get zettles differently
    z_process = ZettelPreProcessor(zettels)

    suggested_keywords = z_process.run()

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
