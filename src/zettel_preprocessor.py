import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


class ZettelPreProcessor:

	def init_zettels(self, zet):
		global zettels
		zettels = zet

	def process_zettels(self):
		tokens = ZettelPreProcessor.tokenizer(self)
		pos_tagged_tokens = ZettelPreProcessor.pos_tagger(self, tokens)
		filtered_words = ZettelPreProcessor.remove_stop_words(self, pos_tagged_tokens)
		# stemmer types: 'porter', 'lancaster', 'snowball'
		stemmed_tokens = ZettelPreProcessor.stemmer(self, filtered_words, 'lancaster')
		lemmatized_tokens = ZettelPreProcessor.lematizer(self, stemmed_tokens)
		return lemmatized_tokens

	def tokenizer(self):
		tokens = []
		for zettel in zettels:
			new_list = re.split('\W+', str(zettel))
			for token in new_list:
				tokens.append(token)
		tokens = list(filter(None, tokens))
		return tokens

	def create_unique_corpus(self, tokens):
		unique_corpus = []
		for word in tokens:
			if word not in unique_corpus:
				unique_corpus.append(word)
		unique_corpus = list(filter(None, unique_corpus))
		unique_corpus.sort()
		return unique_corpus

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
		for zettel in zettels:
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
		n_grams = []
		for index in range(len(tokens)-n+1):
			pair = tokens[index:index+n]
			split = pair[0] + " " + pair[1]
			n_grams.append(split)
		return n_grams

	def remove_stop_words(self, tokens):
		filtered_words = []
		stop_words = set(stopwords.words('english'))
		for word in tokens:
			if word[0].lower() in stop_words:
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
		stemmed_tokens = []
		for word in tokens:
			stemmed_tokens.append([stemmer.stem(word[0]), word[1]])
		return stemmed_tokens

	def pos_tagger(self, tokens):
		tags = nltk.pos_tag(tokens)
		tokens_with_pos_tags = []
		for pair in tags:
			if pair[1].startswith('J'):
				tokens_with_pos_tags.append([pair[0], 'a'])
			elif pair[1].startswith('V'):
				tokens_with_pos_tags.append([pair[0], 'v'])
			elif pair[1].startswith('N'):
				tokens_with_pos_tags.append([pair[0], 'n'])
			elif pair[1].startswith('R'):
				tokens_with_pos_tags.append([pair[0], 'r'])
		return tokens_with_pos_tags

	def lematizer(self, tokens_pos):
		lemmatized_tokens = []
		lemmatizer = WordNetLemmatizer()
		for word in tokens_pos:
			lemmatized_tokens.append(lemmatizer.lemmatize(word[0], word[1]))
		return lemmatized_tokens

	def create_count_dictionary(self, tokens):
		word_count_dict = {}
		for word in tokens:
			if word in word_count_dict:
				word_count_dict[word] += 1
			else:
				word_count_dict[word] = 1
		return word_count_dict

	def create_doc_count_dictionary(self, tokens):
		doc_count_dict = {}
		for zettel in zettels:
			process = ZettelPreProcessor()
			process.init_zettels(zettel)
			lemmatized_tokens = process.process_zettels()
			cur_zettel_dict = {key: 1 for key in lemmatized_tokens}
			word_dict = {}
			for word in tokens:
				if word in word_dict:
					continue
				if word in cur_zettel_dict:
					if word in doc_count_dict:
						doc_count_dict[word] += 1
						word_dict[word] = 1
					else:
						doc_count_dict[word] = 1
						word_dict[word] = 1
		return doc_count_dict
