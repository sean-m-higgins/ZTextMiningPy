import spacy
import os
import numpy as np  # TODO
import weights
import re


class ZettelPreProcessor:

    def __init__(self, zet):
        nlp = spacy.load("en_core_web_sm")
        self.doc = nlp(zet)
        # for token in self.doc:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #           token.shape_, token.is_alpha, token.is_stop)


class ZettelKE:

    def __init__(self, docs):
        z_weights = weights.Weights()
        self.score_weights = z_weights.all_score_weights
        self.pos_score_switch = z_weights.pos_switch
        self.z_area_switch = z_weights.z_area_switch
        self.keyword_n = z_weights.keyword_n
        self.min_keyword_freq = z_weights.min_keyword_freq
        self.docs = docs
        # self.doc_count_dict = self.create_doc_count_dictionary()
        # self.tf_idf_scores = self.tf_idf()
        # self.word_scores = self.create_word_score()
        # self.keyword_scores = self.create_keyword_score()
        # self.pos_scores = {}
        # self.z_area_scores = {}
        # self.create_pos_and_area_score()
        # self.all_scores = self.weight_distribution()
        # self.all_scores_dict = self.get_all_scores_dict()

    def run(self):
        """ Get top n keywords """
        return self.get_keywords()

    def create_count_dictionary(self, tokens):
        """ {word: count} """
        word_count_dict = {}
        for word in tokens:
            word_count_dict.setdefault(word[0], 0)
            word_count_dict[word[0]] += 1
        return word_count_dict

    def create_doc_count_dictionary(self, unique_tokens):  # TODO
        """ {word: doc_count} """
        doc_count_dict = {}
        for zettel in self.lemmatized_tokens:
            for token in zettel:
                for word in unique_tokens:  # TODO check what word is
                    if token[0] == word:
                        doc_count_dict.setdefault(word, 0)
                        doc_count_dict[word] += 1
                        break
        return doc_count_dict

    def tf_idf(self):
        """ tf_idf = tf * idf """
        all_tf_idf = {}  # TODO?
        total_docs = len(self.lemmatized_tokens)
        for zettel in self.lemmatized_tokens:
            total_words = len(zettel)
            count_dict = self.create_count_dictionary(zettel)
            for word in zettel:
                if word[0] not in all_tf_idf:
                    # tf = (count of given word for a given zettel) / (total number of words for given zettel)
                    tf = count_dict[word[0]] / total_words
                    # idf = (total number of documents) / (number of documents containing word)
                    idf = total_docs / self.doc_count_dict[word[0]]
                    tf_idf_value = tf * idf
                    all_tf_idf[word[0]] = tf_idf_value
        return all_tf_idf

    # https://github.com/fabianvf/python-rake/blob/master/RAKE/RAKE.py
    def create_word_score(self):
        """ word_score = deg(word) / freq(word)
            deg(word) = phrase_len - 1 + freq(word) --- uni-gram = 0, bi-gram = 1, tri-gram = 2 """
        word_freq = {}
        word_deg = {}
        for zettel in self.lemmatized_tokens:
            for word in zettel:
                word_list = re.split(" ", word[0])
                word_list_deg = len(word_list) - 1
                for single_word in word_list:
                    word_freq.setdefault(single_word, 0)
                    word_freq[single_word] += 1
                    word_deg.setdefault(single_word, 0)
                    word_deg[single_word] += word_list_deg
        word_score = {}
        for word in word_freq:
            word_deg[word] += word_freq[word]
            word_score.setdefault(word, 0)
            word_score[word] = word_deg[word] / (word_freq[word] * 1.0)
        return word_score

    # https://github.com/fabianvf/python-rake/blob/master/RAKE/RAKE.py
    def create_keyword_score(self):
        """ keyword_score = sum of each word_score in phrase """
        keywords_score = {}
        for zettel in self.lemmatized_tokens:
            for word in zettel:
                if zettel.count(word) >= self.min_keyword_freq:
                    keywords_score.setdefault(word[0], 0)
                    word_list = re.split(" ", word[0])
                    score = 0
                    for new_word in word_list:
                        score += self.word_scores[new_word]
                    keywords_score[word[0]] = score
        return keywords_score

    def create_pos_and_area_score(self):
        """ pos_score = ('NN', .40) ('NNS', .35) ('NNP', .80) ('NNPS', .70) ('NG', .50)
            z_area_score = (title: .80) (summary: .60) (note: 40) """
        for zettel in self.lemmatized_tokens:
            for word in zettel:
                self.pos_scores.setdefault(word[0], 0)
                self.pos_scores[word[0]] = self.pos_score_switch.get(word[1], 0)
                self.z_area_scores.setdefault(word[0], 0)
                self.z_area_scores[word[0]] = self.z_area_switch.get(word[2])

    def weight_distribution(self):
        """ combine all scores together with weights """
        all_scores = []  # TODO
        for zettel in self.lemmatized_tokens:
            scores = []  # TODO
            for word in zettel:
                cur_tf_idf = self.tf_idf_scores[word[0]] / 3  # range: 0-3+  #TODO
                cur_keyword_score = self.keyword_scores[word[0]] / 4  # 0-4+  #TODO
                cur_pos_score = self.pos_scores[word[0]]
                cur_area_score = self.z_area_scores[word[0]]
                cur_total_score = ((cur_tf_idf * self.score_weights[0]) +
                                   (cur_keyword_score * self.score_weights[1]) +
                                   (cur_pos_score * self.score_weights[2]) +
                                   (cur_area_score * self.score_weights[3])) / 4
                scores.append(cur_total_score)  # TODO
            all_scores.append(scores)  # TODO
        return all_scores

    def get_all_scores_dict(self):  # TODO delete? and do in above emthod
        z_index = 0
        new_all_scores = {}
        for zettel in self.lemmatized_tokens:
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

    def get_final_keywords(self, keywords_dict):  # TODO all
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
                        if keywords_dict[word] > keywords_dict[keyword] or keywords_dict[word] == keywords_dict[
                            keyword]:
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
        all_keywords = []  # TODO
        z_index = 0
        for zettel in self.lemmatized_tokens:
            keywords = []  # TODO
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
            cur_sorted = sorted(final_zettel_dict.items(), key=lambda kv: kv[1], reverse=True)  # TODO sort numpy?
            for i in range(self.keyword_n):
                keywords.append(str(cur_sorted[i]))  # TODO
            z_index += 1
            all_keywords.append(keywords)  # TODO
        return all_keywords


def get_zettels_from_clean_directory(directory):
    new_zettels = []
    files = os.listdir(directory)
    for file in files:
        path = directory + "/" + file
        zettel = ""
        lines = open(path).readlines()
        for line in lines:
            zettel += line
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
    # zettels = get_zettels_from_clean_directory(clean_baseball)  # TODO get zettles differently
    zettels = get_zettels_from_clean_directory(movies)

    docs = []
    for zettel in zettels:
        z_preprocess = ZettelPreProcessor(zettel)
        docs.append(z_preprocess.doc)

    z_ke = ZettelKE(docs)
    # suggested_keywords = z_ke.run()
    #
    # index = 0
    # for zettel in suggested_keywords:
    #     print("\nSuggested Keywords for Zettel " + str(index) + ": ")
    #     inner_i = 1
    #     for item in zettel:
    #         print(str(inner_i) + ": ")
    #         print(item)
    #         inner_i += 1
    #     index += 1

    print("Done.")
    print(datetime.datetime.now())