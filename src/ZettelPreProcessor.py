import numpy as np
import re
from nltk.corpus import stopwords


class ZettelPreProcessor:
	zettels = []

	def init_zettels(self, zet):
		ZettelPreProcessor.zettels = zet

	def tokenizer(self):
		tokens = []
		for zettel in ZettelPreProcessor.zettels:
			list = re.split("\W+", zettel.lower())
			for token in list:
				tokens.append(token)
		return tokens

	def create_unique_corpus(self, tokens):
		return np.unique(tokens)

	def create_unique_tag_corpus(self, tokens):
		unique_tag_corpus = []
		lock = 0
		for word in tokens:
			if word == "tags":
				lock = 1
				continue
			if word == "cite":
				lock = 0
			while lock == 1:
				unique_tag_corpus.append(word)
				break
		return np.unique(unique_tag_corpus)

	def create_count_matrix(self, unique_corpus):
		count_matrix = []
		for zettel in ZettelPreProcessor.zettels:
			count = ZettelPreProcessor.get_word_count(ZettelPreProcessor, zettel, unique_corpus)
			count_matrix.append(count)
		return count_matrix

	def get_word_count(self, zettel, unique_corpus):
		new_unique_corpus = unique_corpus
		count = np.zeros(new_unique_corpus.size)
		split_zettel = re.split("\W+", zettel.lower())
		for word in split_zettel:
			new_iter = iter(unique_corpus)
			i = 0
			for new_word in new_iter:
				if word == new_word:
					count[i] = 1
				i += 1
		return count

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

	def remove_stop_words(self, tokens):
		filtered_words = []
		for word in tokens:
			if word in set(stopwords.words("english")):
				continue
			else:
				filtered_words.append(word)
		filter(None, filtered_words)
		return filtered_words