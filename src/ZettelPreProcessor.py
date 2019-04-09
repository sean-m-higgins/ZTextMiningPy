import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer


class ZettelPreProcessor:
	zettels = []

	def init_zettels(self, zet):
		ZettelPreProcessor.zettels = zet

	def process_zettels(self):
		tokens = ZettelPreProcessor.tokenizer(self)
		filtered_words = ZettelPreProcessor.remove_stop_words(self, tokens)
		# stemmer types: 'porter', 'lancaster', 'snowball'
		stemmed_tokens = ZettelPreProcessor.stemmer(self, filtered_words, 'lancaster')
		pos_tagged_tokens = ZettelPreProcessor.pos_tagger(self, stemmed_tokens)
		lemmatized_tokens = ZettelPreProcessor.lematizer(self, pos_tagged_tokens)
		return lemmatized_tokens

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
			if word == "tags":
				lock = 1
				continue
			if word == "cite":
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
					count[i] = count[i] + 1
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

	def stemmer(self, tokens, stemmer_type):
		switch = {
			'porter': PorterStemmer(),
			'lancaster': LancasterStemmer(),
			'snowball': SnowballStemmer('english'),
		}
		stemmer = switch.get(stemmer_type)
		new_tokens = []
		for word in tokens:
			new_tokens.append(stemmer.stem(word))
		return new_tokens

	def pos_tagger(self, tokens):
		tags = nltk.pos_tag(tokens)
		new_tags = []
		for pair in tags:
			if pair[1].startswith('J'):
				new_tags.append([pair[0], 'a'])
			elif pair[1].startswith('V'):
				new_tags.append([pair[0], 'v'])
			elif pair[1].startswith('N'):
				new_tags.append([pair[0], 'n'])
			elif pair[1].startswith('R'):
				new_tags.append([pair[0], 'r'])
		return new_tags

	def lematizer(self, tokens_pos):
		new_tokens = []
		lemmatizer = WordNetLemmatizer()
		for word in tokens_pos:
			new_tokens.append(lemmatizer.lemmatize(word[0], word[1]))
		return new_tokens

	def create_count_dictionary(self, tokens):
		word_dict = {}
		for word in tokens:
			if word in word_dict:
				word_dict[word] += 1
			else:
				word_dict[word] = 1
		return word_dict

	def create_doc_count_dictionary(self, tokens):
		doc_count_dict = {}
		zettels = ZettelPreProcessor.zettels
		for zettel in zettels:
			ZettelPreProcessor.init_zettels(self, zettel)
			cur_zettel = ZettelPreProcessor.process_zettels(self)
			z_dict = {key: 1 for key in cur_zettel}
			word_dict = {}
			for word in tokens:
				# if word was already evaluated... break
				if word in word_dict:
					continue
				# if word is in current zettel...
				if word in z_dict:
					if word in doc_count_dict:
						doc_count_dict[word] += 1
						word_dict[word] = 1
					else:
						doc_count_dict[word] = 1
						word_dict[word] = 1
		return doc_count_dict


