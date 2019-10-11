import nltk
import re


class ZettelPreProcessor:

    def __init__(self, zet):
        self.zettel = zet
        self.given_tags = []
        self.tokens = self.tokenizer()
        stopwords_file = open("../docs/data/processedData/stopWords/zettelStopWords.txt", "r")
        self.stop_words = [line[:-1] for line in stopwords_file.readlines()]  # removes /n from each line
        self.filtered_words = self.remove_stop_words()
        self.pos_tagged_tokens = self.pos_tagger()
        self.filter_n_grams(self.create_n_gram(2))
        self.filter_n_grams(self.create_n_gram(3))
        self.count_dict = self.create_count_dictionary()

    def tokenizer(self):
        """ Split zettels by word """
        all_tokens = []
        index = 0  #TODO remove
        # for section, index in zip(self.zettel, range(len(self.zettel))):  #TODO
        #     if index == 3:
        #         self.given_tags = re.split(";", section)  #TODO make sure to use ';' 
        for section in self.zettel:
            if index == len(self.zettel) - 1:
                self.given_tags = re.split(";", section)
            else:
                tokens = re.split('\W+', section)
                tokens = list(filter(None, tokens))
                all_tokens.append(tokens)
            index += 1  #TODO remove
        return all_tokens

    def remove_stop_words(self):
        """ Remove any words included in stop words list """
        all_filtered_words = []
        for section in self.tokens:
            new_section = []
            for word in section:
                if word not in self.stop_words and word.lower() not in self.stop_words and word != '':
                    new_section.append(word)
            all_filtered_words.append(new_section)
        return all_filtered_words

    def pos_tagger(self):
        """ Tag each word with its part of speech """
        all_tokens_with_pos_tags = []
        aread_id = 0
        for section in self.filtered_words:
            tags = nltk.pos_tag(section)
            for word in tags:
                if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'NG']:  # NG = n_gram
                    if word[1].startswith('N'):
                        all_tokens_with_pos_tags.append([word[0], word[1], aread_id])
            aread_id += 1
        return all_tokens_with_pos_tags

    def create_n_gram(self, n):
        """ Create list of candidate n-grams """
        all_n_grams = []
        for section in self.tokens:
            for index in range(len(section) - n + 1):
                new_set = section[index:index + n]
                if (new_set[0].lower() in self.stop_words) or (
                        new_set[n - 1].lower() in self.stop_words):  # skip if begin/end with stop_word
                    continue
                split_str = ""
                for i in range(n):
                    split_str += new_set[i]
                    if i < n - 1:
                        split_str += " "
                all_n_grams.append(split_str)
        return all_n_grams

    def filter_n_grams(self, n_grams):
        """ remove infrequent n_grams and add frequent n_grams to respective zettel in tokens """
        all_n_grams = []
        for gram in n_grams:
            if n_grams.count(gram) >= 2:
                if gram not in all_n_grams:
                    split_gram = re.split(" ", gram)
                    tags = nltk.pos_tag(split_gram)
                    if tags[0][1][0] == 'V' or tags[len(split_gram)-1][1][0] == 'V':
                        continue
                    all_n_grams.append(gram)
        for gram in all_n_grams:
            self.pos_tagged_tokens.append([gram, 'NG', 3])

    def create_count_dictionary(self):
        """ {word: count} """
        word_count_dict = {}
        for word in self.pos_tagged_tokens:
            word_count_dict.setdefault(word[0], 0)
            word_count_dict[word[0]] += 1
        return word_count_dict
