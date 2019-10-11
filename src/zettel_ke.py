import zettel_preprocess
import re


class ZettelKE:

    def __init__(self, tokens, count_dicts, given_tags):
        self.score_weights = [0.90, 0.50, 0.55, 0.25]  # tf_idf, keyword_score, pos_score, area_score [0.60, 0.30, 0.05, 0.05]
        self.pos_score_switch = {
            'NN': 0.30,
            'NNS': 0.20,
            'NNP': 0.70,
            'NNPS': 0.60,
            'NG': 0.90
        }
        self.z_area_switch = {
            0: 0.90,
            1: 0.60,
            2: 0.30,
            3: 0.90  # n-gram
        }
        self.keyword_n = 7  #TODO
        self.min_keyword_freq = 1
        self.tokens = tokens
        self.count_dicts = count_dicts
        self.given_tags = given_tags
        self.unique_tags = self.create_unique_tags()
        self.unique_tokens = self.create_unique_corpus()
        self.doc_count_dict = self.create_doc_count_dictionary()
        self.tf_idf_scores = self.tf_idf()  
        self.word_scores = self.create_word_score()  
        self.keyword_scores = self.create_keyword_score()  
        self.all_scores = self.weight_distribution()

    def run(self):
        """ Get top n keywords """
        return self.get_keywords()

    def create_unique_corpus(self):
        """ Create distinct set of words """
        token_set = []
        for zettel in self.tokens:
            for word in zettel:
                if word[0] not in token_set:
                    token_set.append(word[0])
        return token_set

    def create_unique_tags(self):
        """ Create distinct set of given_tags """
        tags_set = []
        for zettel in self.given_tags:
            for word in zettel:
                if word not in tags_set:
                    tags_set.append(word)
        return tags_set

    def create_doc_count_dictionary(self):
        """ {word: doc_count} """
        doc_count_dict = {}
        for zettel in self.tokens:
            for word in self.unique_tokens:
                for token in zettel: 
                    if token[0] == word:
                        doc_count_dict.setdefault(word, 0)
                        doc_count_dict[word] += 1
                        break
        return doc_count_dict

    def tf_idf(self):  
        """ tf_idf = tf * idf """
        all_tf_idf = {}
        total_docs = len(self.tokens)
        for zettel, count_dict in zip(self.tokens, self.count_dicts):
            total_words = len(zettel)
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
                        score += self.word_scores[new_word]     #TODO other scores? replace word score with tf_idf?
                    keywords_score[word[0]] = score
        return keywords_score

    def weight_distribution(self):
        """ combine all scores together with weights """
        all_scores = []
        for zettel in self.tokens:
            zettel_scores = {}
            for word in zettel:
                tag_score = 0
                if word[0] in self.unique_tags:  #if word is in the set of given tags, add 1 to total score
                    tag_score = 1
                cur_tf_idf = self.tf_idf_scores[word[0]]
                cur_keyword_score = self.keyword_scores[word[0]]
                cur_pos_score = self.pos_score_switch.get(word[1])
                cur_area_score = self.z_area_switch.get(word[2])
                cur_total_score = ((cur_tf_idf * self.score_weights[0]) +
                                   (cur_keyword_score * self.score_weights[1]) +
                                   (cur_pos_score * self.score_weights[2]) +
                                   (cur_area_score * self.score_weights[3])) / 4
                zettel_scores[word[0]] = cur_total_score + tag_score
            all_scores.append(zettel_scores)
        return all_scores

    def get_final_keywords(self, cur_keywords_dict):
        """ Return a list of all keywords without matches, by keeping the higher scored match """
        all_matches = self.get_matches(cur_keywords_dict)
        new_check = self.check_matches(all_matches)
        top_matches = self.get_top_matches(all_matches, cur_keywords_dict)
        while new_check:
            top_matches = self.get_top_matches(all_matches, cur_keywords_dict)
            new_all_matches = self.get_matches(top_matches)
            new_check = self.check_matches(new_all_matches)
            all_matches = new_all_matches
        return top_matches

    def get_top_matches(self, all_matches, cur_keywords_dict):
        """ Return dictionary of keywords with the highest score out of its matches """
        top_matches = {}
        for match_list in all_matches:
            top_word = match_list[0]
            for word in match_list:
                if cur_keywords_dict[top_word] < cur_keywords_dict[word]:
                    top_word = word
            top_matches[top_word] = cur_keywords_dict[top_word]
        return top_matches

    def get_matches(self, word_dict):
        """ Return a list of words and their matches, if no match the row is simply the word """
        all_matches = []
        word_index = 0
        for word in word_dict:
            new_matches = []
            new_matches.append(word)
            keyword_index = 0
            for keyword in word_dict:
                if word_index != keyword_index:
                    if word_dict[keyword] == word_dict[word]:
                        continue
                    keyword_index += 1
                    if word.lower() == keyword.lower():
                        new_matches.append(keyword)
                    elif self.split_check(re.split(" ", word), re.split(" ", keyword)):
                        new_matches.append(keyword)
                else:
                    keyword_index += 1
            all_matches.append(new_matches)
            word_index += 1
        return all_matches

    def split_check(self, split_word, split_keyword):
        """ if any of the words in split_word match any of the words in split_keyword, return true, else false """
        check = False
        for item in split_word:
            for key_item in split_keyword:
                if item.lower() == key_item.lower():
                    check = True
        return check

    def check_matches(self, all_matches):
        """ Returns True if there are matches needed to be resolved """
        check = False
        for match in all_matches:
            if len(match) != 1: 
                check = True
        return check

    def get_keywords(self):
        """ get top n keywords based on total score for each zettel """
        all_keywords = []
        for cur_zettel_dict, cur_tags in zip(self.all_scores, self.given_tags):
            for tag in cur_tags:
                cur_zettel_dict[tag] = 100
            cur_sorted = sorted(cur_zettel_dict.items(), key=lambda kv: kv[1], reverse=True)
            new_dict = {}
            for i in range(self.keyword_n + len(cur_tags)):
                new_dict[cur_sorted[i][0]] = cur_sorted[i][1]
            final_zettel_dict = self.get_final_keywords(new_dict)
            keywords = []  #TODO change
            for item in final_zettel_dict:
                keywords.append(str(item) + ": " + str(final_zettel_dict[item]))  #TODO change what to be added to not a str? or change str? make sure to match the given_tags
            all_keywords.append(keywords)
        return all_keywords
