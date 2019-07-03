from src import distance, cluster
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords # stopwords.words('english')
import os


class ZettelPreProcessor:

	def init_zettels(self, zet):
		self.zettels = zet
		self.tokens = self.tokenizer()
		sw_file = open("/Users/SeanHiggins/ZTextMiningPy/docs/data/processedData/stopWords/zettelStopWords.txt", "r")
		self.stop_words = re.split("\n", sw_file.read())  #TODO possibly remove title, note... from file
		self.filtered_words = self.remove_stop_words()
		self.pos_tagged_tokens = self.pos_tagger()
		self.lemmatized_tokens = self.lemmatizer()
		#self.stemmed_tokens = self.stemmer('lancaster')  #stemmer types: 'porter', 'lancaster', 'snowball'  TODO remove?
		self.unique_corpus = self.create_unique_corpus()
		self.bi_gram = self.create_n_gram(2)
		self.tri_gram = self.create_n_gram(3)
		#self.unique_count_matrix = self.create_count_matrix() TODO remove?
		#self.unique_tag_corpus = self..create_unique_tag_corpus(tags)  #TODO get tokens from an array and parse...
		#self.tag_boolean_matrix = self.create_boolean_tag_matrix(unique_tag_corpus)
		#self.tag_count_matrix = self.create_unique_corpus(unique_tag_corpus)

	def tokenizer(self):
		all_tokens = []
		for zettel in self.zettels:
			tokens = []
			new_list = re.split('\W+', str(zettel))
			for token in new_list:
				tokens.append(token)
			tokens = list(filter(None, tokens))
			all_tokens.append(tokens)
		return all_tokens

	def remove_stop_words(self):
		all_filtered_words = []
		for zettel in self.tokens:
			filtered_words = []
			for word in zettel:
				if word.lower() in self.stop_words:
					continue
				else:
					filtered_words.append(word)
			filter(None, filtered_words)
			all_filtered_words.append(filtered_words)
		return all_filtered_words

	def pos_tagger(self):
		all_tokens_with_pos_tags = []
		for zettel in self.filtered_words:
			tags = nltk.pos_tag(zettel)
			tokens_with_pos_tags = []
			for word in tags:
				if word[1].startswith('J'):
					tokens_with_pos_tags.append([word[0], word[1], 'a'])
				elif word[1].startswith('V'):
					tokens_with_pos_tags.append([word[0], word[1], 'v'])
				elif word[1].startswith('N'):
					tokens_with_pos_tags.append([word[0], word[1], 'n'])
				elif word[1].startswith('R'):
					tokens_with_pos_tags.append([word[0], word[1], 'r'])
			all_tokens_with_pos_tags.append(tokens_with_pos_tags)
		return all_tokens_with_pos_tags

	def lemmatizer(self):
		all_lemmatized_tokens = []
		lemmatizer = WordNetLemmatizer()
		for zettel in self.pos_tagged_tokens:
			lemmatized_tokens = []
			for word in zettel:
				lemmatized_tokens.append([lemmatizer.lemmatize(word[0], word[2]), word[1]])
			all_lemmatized_tokens.append(lemmatized_tokens)
		return all_lemmatized_tokens

	def stemmer(self, stemmer_type):
		switch = {
			'porter': PorterStemmer(),
			'lancaster': LancasterStemmer(),
			'snowball': SnowballStemmer('english'),
		}
		stemmer = switch.get(stemmer_type)
		all_stemmed_tokens = []
		for zettel in self.pos_tagged_tokens:
			stemmed_tokens = []
			for word in zettel:
				stemmed_tokens.append([stemmer.stem(word[0]), word[2]])
			all_stemmed_tokens.append(stemmed_tokens)
		return all_stemmed_tokens

	def create_unique_corpus(self):
		unique_corpus = []
		for zettel in self.lemmatized_tokens:
			for word in zettel:
				if word[0] not in unique_corpus:
					unique_corpus.append(word[0])
		unique_corpus = list(filter(None, unique_corpus))
		unique_corpus.sort()
		return unique_corpus

	# def create_unique_tag_corpus(self):  #TODO fix once tags are recieved correctly
	# 	unique_tag_corpus = []
	# 	lock = 0
	# 	for word in self.tokens:
	# 		if word == "tags":
	# 			lock = 1
	# 			continue
	# 		if word == "cite":
	# 			lock = 0
	# 		while lock == 1:
	# 			if word not in unique_tag_corpus:
	# 				unique_tag_corpus.append(word)
	# 			break
	# 	unique_tag_corpus = list(filter(None, unique_tag_corpus)).sort()
	# 	return unique_tag_corpus

	def create_count_matrix(self):  #TODO check
		count_matrix = []
		for zettel in self.zettels:
			count = ZettelPreProcessor.get_word_count(self, zettel)
			count_matrix.append(count)
		return count_matrix

	def get_word_count(self, zettel):  #TODO check
		new_unique_corpus = self.unique_corpus
		count = np.zeros(len(new_unique_corpus))
		split_zettel = re.split("\W+", str(zettel).lower())
		for word in split_zettel:
			new_iter = iter(self.unique_corpus)
			i = 0
			for new_word in new_iter:
				if word == new_word:
					count[i] = count[i] + 1
				i += 1
		return count.tolist()

	# def create_boolean_tag_matrix(self): 	#TODO
	# 	unique_tag_count_matrix = ZettelPreProcessor.create_count_matrix(self)
	# 	tag_boolean_matrix = []
	# 	for row in unique_tag_count_matrix:
	# 		inner_boolean = []
	# 		for count in row:
	# 			if count == 0:
	# 				inner_boolean.append(False)
	# 			else:
	# 				inner_boolean.append(True)
	# 		tag_boolean_matrix.append(inner_boolean)
	# 	return tag_boolean_matrix

	def create_n_gram(self, n):
		all_n_grams = []
		for zettel in self.tokens:
			n_grams = []
			for index in range(len(zettel)-n+1):
				set = zettel[index:index+n]
				if (set[0].lower() in self.stop_words) or (set[n-1].lower() in self.stop_words): #skip if begin/end with stop_word
					continue
				split = ""
				for i in range(n):
					split += set[i]
					if i < n-1:
						split = split + " "
				n_grams.append(split)
			all_n_grams.append(n_grams)
		return all_n_grams

	# def create_count_dictionary(self, tokens): #TODO replaced?
	# 	word_count_dict = {}
	#	for word in tokens:
	#      	word_count_dict.setdefault(word, 0)
    #		word_count_dict[word] += 1
    #	return word_count_dict
	#
	# def create_doc_count_dictionary(self):
	# 	doc_count_dict = {}
	# 	for zettel in self.zettels:
	# 		process = ZettelPreProcessor()
	# 		process.init_zettels(zettel)
	# 		cur_zettel_dict = {key: 1 for key in process.lemmatized_tokens}
	# 		word_dict = {}
	# 		for word in self.lemmatized_tokens:
	# 			if word in word_dict:
	# 				continue
	# 			if word in cur_zettel_dict:
	# 				if word in doc_count_dict:
	# 					doc_count_dict[word] += 1
	# 					word_dict[word] = 1
	# 				else:
	# 					doc_count_dict[word] = 1
	# 					word_dict[word] = 1
	# 	return doc_count_dict

	def get_zettels_from_directory(self, directory):
		new_zettels = []
		files = os.listdir(directory)
		for file in files:
			path = directory + "/" + file
			contents = [str([line.rstrip() for line in open(path)])]
			new_zettels.append(contents)
		return new_zettels

	def get_document_word_counts(self):
		i = 0
		counts = []
		for zettel in self.tokens:
			counts.append(0)
			for word in zettel:
				counts[i] = counts[i] + 1
			i += 1
		return counts

if __name__ == "__main__":
	baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
	bibs = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/bibs"
	examples = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/examples"
	rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"

	process = ZettelPreProcessor()
	zettels = process.get_zettels_from_directory(baseball)
	process.init_zettels(zettels)

	distance = distance.Distance()
	distance_type = 'euclidean'
	matrix = np.array(distance.get_distance_matrix(process.unique_count_matrix, distance_type))
	tf_idf = distance.tf_idf(zettels)

	cluster = cluster.Cluster()
	# hierarchical_cluster = cluster.hclust(matrix, distance_type)
	# hierarchical_cluster = cluster.hclust(tf_idf, 'tf idf')
	# k_means = cluster.k_means(matrix, distance_type)
