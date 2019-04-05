import numpy as np
import re


class ZettelPreProcessor:
	zettels = []

	def init_zettels(self, zet):
		ZettelPreProcessor.zettels = zet

	def tokenizer(self):
		tokens = []
		for zettel in ZettelPreProcessor.zettels:
			new_list = re.split('\W+', str(zettel).lower())
			for token in new_list:
				tokens.append(token)
		tokens = list(filter(None, tokens))
		return tokens

	def create_unique_corpus(self, tokens):
		unique = []
		for word in tokens:
			if word not in unique:
				unique.append(word)
		unique = list(filter(None, unique))
		unique.sort()
		return unique

	def create_unique_tag_corpus(self, tokens):
		unique_tag_corpus = []
		lock = 0
		for word in tokens:
			if word == "ntags": #TODO
				lock = 1
				continue
			if word == "ncite": #TODO
				lock = 0
			while lock == 1:
				if word not in unique_tag_corpus:
					unique_tag_corpus.append(word)
				break
		unique_tag_corpus = list(filter(None, unique_tag_corpus))
		unique_tag_corpus.sort()
		return unique_tag_corpus

	def create_count_matrix(self, unique_corpus):
		count_matrix = []
		for zettel in ZettelPreProcessor.zettels:
			count = ZettelPreProcessor.get_word_count(self, zettel, unique_corpus)
			count_matrix.append(count)
		return count_matrix

	def get_word_count(self, zettel, unique_corpus):
		new_unique_corpus = unique_corpus
		count = np.zeros(len(new_unique_corpus))
		split_zettel = re.split("\W+", str(zettel).lower())
		for word in split_zettel:
			new_iter = iter(unique_corpus)
			i = 0
			for new_word in new_iter:
				if word == new_word:
					count[i] = 1
				i += 1
		return count.tolist()

	def create_boolean_tag_matrix(self, unique_tags):
		unique_tag_count_matrix = ZettelPreProcessor.create_count_matrix(self, unique_tags)
		tag_boolean_matrix = []
		for row in unique_tag_count_matrix:
			inner_boolean = []
			for count in row:
				if count == 0:
					inner_boolean.append(False)
				else:
					inner_boolean.append(True)
			tag_boolean_matrix.append(inner_boolean)
		return tag_boolean_matrix

	def create_n_gram(self, tokens, n):
		n_gram = []
		for i in range(len(tokens)-n+1):
			n_gram.append(tokens[i:i+n])
		return n_gram

	def get_stop_words(self):
		stopwords = "/Users/SeanHiggins/ZTextMiningPy/docs/data/processedData/stopWords/sparkStopWords.txt"
		file = open(stopwords, "r")
		contents = file.read()
		return contents

	def remove_stop_words(self, tokens):
		filtered_words = []
		stopwords = ZettelPreProcessor.get_stop_words(self)
		for word in tokens:
			if word in stopwords:
				continue
			else:
				filtered_words.append(word)
		filter(None, filtered_words)
		return filtered_words

	# TODO def stemmer:

	# TODO def lematizer:
