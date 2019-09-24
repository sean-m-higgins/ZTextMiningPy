import zettel_preprocess
import weights
import os
import re


class ZettelKE:

    def __init__(self, tokens):
        z_weights = weights.Weights()  #TODO remove weights class and set values here
        self.score_weights = z_weights.all_score_weights
        self.pos_score_switch = z_weights.pos_switch
        self.z_area_switch = z_weights.z_area_switch
        self.keyword_n = z_weights.keyword_n
        self.min_keyword_freq = z_weights.min_keyword_freq
        self.tokens = tokens
        self.unique_tokens = self.create_unique_corpus()  #TODO store in database
        self.doc_count_dict = self.create_doc_count_dictionary()  #TODO store in database and get and set here
        self.tf_idf_scores = self.tf_idf()  #TODO get and set scores
        self.word_scores = self.create_word_score()  #TODO get and set scores
        self.keyword_scores = self.create_keyword_score()  #TODO get and set scores
        self.all_scores = self.weight_distribution()

    def run(self):
        """ Get top n keywords """
        return self.get_keywords()

    def create_unique_corpus(self):  #TODO set up for tags; if cur zettel's suggested keyword is in this corpus, add to suggestions after sorting.
        """ Create distinct set of words """
        token_set = []
        for zettel in self.tokens:
            for word in zettel:
                if word[0] not in token_set:
                    token_set.append(word[0])
        return token_set

    def create_count_dictionary(self, cur_tokens):
        """ {word: count} """
        word_count_dict = {}
        for word in cur_tokens:
            word_count_dict.setdefault(word[0], 0)
            word_count_dict[word[0]] += 1
        return word_count_dict

    def create_doc_count_dictionary(self):
        """ {word: doc_count} """
        doc_count_dict = {}
        for zettel in self.tokens:
            for token in zettel:  #TODO
                for word in self.unique_tokens:
                    if token[0] == word:
                        doc_count_dict.setdefault(word, 0)
                        doc_count_dict[word] += 1
                        break
        return doc_count_dict

    def tf_idf(self):  #TODO
        """ tf_idf = tf * idf """
        all_tf_idf = {}
        total_docs = len(self.tokens)
        for zettel in self.tokens:
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
        for zettel in self.tokens:
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
        for zettel in self.tokens:
            for word in zettel:
                if zettel.count(word) >= self.min_keyword_freq:
                    keywords_score.setdefault(word[0], 0)
                    word_list = re.split(" ", word[0])
                    score = 0
                    for new_word in word_list:
                        score += self.word_scores[new_word]     #TODO other scores? replace word score with tf_idf?  #remove weight distribution later and combine scores here based on individual words?
                    keywords_score[word[0]] = score
        return keywords_score

    def weight_distribution(self):
        """ combine all scores together with weights """
        all_scores = []
        for zettel in self.tokens:
            zettel_scores = {}
            for word in zettel:
                cur_tf_idf = self.tf_idf_scores[word[0]]
                cur_keyword_score = self.keyword_scores[word[0]]
                cur_pos_score = self.pos_score_switch.get(word[1])
                cur_area_score = self.z_area_switch.get(word[2])
                cur_total_score = ((cur_tf_idf * self.score_weights[0]) +
                                   (cur_keyword_score * self.score_weights[1]) +
                                   (cur_pos_score * self.score_weights[2]) +
                                   (cur_area_score * self.score_weights[3])) / 4
                zettel_scores[word[0]] = cur_total_score
            all_scores.append(zettel_scores)
        return all_scores

    def split_check(self, split_word, split_keyword):
        """ if any of the words in split_word match any of the words in split_keyword, return true, else false """
        check = False
        for item in split_word:
            for key_item in split_keyword:
                if item.lower() == key_item.lower():
                    check = True
        return check

    def get_final_keywords(self, cur_keywords_dict):
        """ Filter out any keywords that appear more than once, choosing the higher scored duplicate """
        new_keywords_dict = {}
        word_index = 0
        black_list = []
        for word in cur_keywords_dict:
            keyword_index = 0
            word_score = cur_keywords_dict[word]
            for keyword in cur_keywords_dict:
                if word_index != keyword_index:
                    keyword_score = cur_keywords_dict[keyword]
                    keyword_index += 1
                    if word.lower() == keyword.lower():  # if two words match, take higher scored phrase
                        if word_score >= keyword_score:
                            if keyword not in black_list:
                                black_list.append(keyword)
                            if word not in new_keywords_dict:
                                new_keywords_dict[word] = word_score
                                break
                            break
                        else:
                            if word not in black_list:
                                black_list.append(word)
                            if keyword not in new_keywords_dict:
                                new_keywords_dict[keyword] = keyword_score
                                break
                            break
                    elif self.split_check(re.split(" ", word), re.split(" ", keyword)):
                        if word_score >= keyword_score:
                            if keyword not in black_list:
                                black_list.append(keyword)
                            if word not in new_keywords_dict:
                                new_keywords_dict[word] = word_score
                                break
                            break
                        else:
                            if word not in black_list:
                                black_list.append(word)
                            if keyword not in new_keywords_dict:
                                new_keywords_dict[keyword] = keyword_score
                                break
                            break
                else:
                    keyword_index += 1
            if word not in black_list:
                if word not in new_keywords_dict:
                    new_keywords_dict[word] = word_score
            word_index += 1
        return new_keywords_dict

    def get_keywords(self):
        """ get top n keywords based on total score for each zettel """
        all_keywords = []
        for cur_zettel_dict in self.all_scores:
            keywords = []
            final_zettel_dict = self.get_final_keywords(cur_zettel_dict)
            cur_sorted = sorted(final_zettel_dict.items(), key=lambda kv: kv[1], reverse=True)
            for i in range(self.keyword_n):
                keywords.append(str(cur_sorted[i]))  #TODO change what to be added to not a str? or change sr?
            all_keywords.append(keywords)
        return all_keywords


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
    movies = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/movies"
    clean_baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"

    import datetime
    print(datetime.datetime.now())

    # zettels = get_zettels_from_clean_directory(clean_baseball)
    zettels = get_zettels_from_clean_directory(movies)  # TODO get zettles differently

    tokens = []
    for zettel in zettels:
        z_preprocess = zettel_preprocess.ZettelPreProcessor(zettel)
        tokens.append(z_preprocess.pos_tagged_tokens)

    z_ke = ZettelKE(tokens)
    suggested_keywords = z_ke.run()

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
