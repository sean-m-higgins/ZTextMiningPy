from src import distance, cluster
import numpy as np
import re
import nltk
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.lancaster import LancasterStemmer
# from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
# from nltk.corpus import stopwords # stopwords.words('english')
import os
import threading
import concurrent.futures
from queue import Queue


class ZettelPreProcessor:

	def init_zettels(self, zet):
		self.lock = threading.Lock()
		self.lemmatizer = WordNetLemmatizer()
		self.zettels = zet
		self.tokens = self.tokenizer()
		sw_file = open("/Users/SeanHiggins/ZTextMiningPy/docs/data/processedData/stopWords/zettelStopWords.txt", "r")
		self.stop_words = [line[:-1] for line in sw_file.readlines()] #TODO possibly remove title, note... from file
		self.filtered_words = self.remove_stop_words()
		self.pos_tagged_tokens = self.pos_tagger()
		self.lemmatized_tokens = self.create_lemmatized_tokens()
		thread_1 = threading.Thread(self.get_bi_gram(), args=(1,))
		thread_1.start()
		thread_2 = threading.Thread(self.get_tri_gram(), args=(2,))
		thread_2.start()
		thread_1.join()
		thread_2.join()
		#self.stemmed_tokens = self.stemmer('lancaster')  #stemmer types: 'porter', 'lancaster', 'snowball'  TODO remove?
		#self.unique_count_matrix = self.create_count_matrix()
		#self.unique_tag_corpus = self..create_unique_tag_corpus(tags)  #TODO get tokens from an array and parse...
		#self.tag_boolean_matrix = self.create_boolean_tag_matrix(unique_tag_corpus)
		#self.tag_count_matrix = self.create_unique_corpus(unique_tag_corpus)

	def tokenizer(self):
		all_tokens = []
		self.token_q = Queue()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			for num in range(5):
				executor.submit(self.get_tokens, num)
		results = {}
		while not self.token_q.empty():
			temp_result = self.token_q.get()
			results[temp_result[1]] = temp_result[0]
		for index in range(5):
			for zettel in results[index]:
				all_tokens.append(zettel)
		return all_tokens

	def get_tokens(self, thread_index):
		switch = {
			0: self.zettels[:(len(self.zettels)//5)],
			1: self.zettels[(len(self.zettels)//5): (len(self.zettels)//5)*2],
			2: self.zettels[(len(self.zettels)//5)*2: (len(self.zettels)//5)*3],
			3: self.zettels[(len(self.zettels)//5)*3: (len(self.zettels)//5)*4],
			4: self.zettels[(len(self.zettels)//5)*4: (len(self.zettels)//5)*5],
		}
		new_tokens = []
		for zettel in switch.get(thread_index):
			tokens = re.split('\W+', str(zettel))
			tokens = list(filter(None, tokens))
			new_tokens.append(tokens)
		self.token_q.put([new_tokens, thread_index])


	def remove_stop_words(self):
		all_filtered_words = []
		self.stop_word_q = Queue()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			for num in range(5):
				executor.submit(self.get_stop_words, num)
		results = {}
		while not self.stop_word_q.empty():
			temp_result = self.stop_word_q.get()
			results[temp_result[1]] = temp_result[0]
		for index in range(5):
			for zettel in results[index]:
				all_filtered_words.append(zettel)
		return all_filtered_words

	def get_stop_words(self, thread_index):
		switch = {
			0: self.tokens[:(len(self.tokens)//5)],
			1: self.tokens[(len(self.tokens)//5): (len(self.tokens)//5)*2],
			2: self.tokens[(len(self.tokens)//5)*2: (len(self.tokens)//5)*3],
			3: self.tokens[(len(self.tokens)//5)*3: (len(self.tokens)//5)*4],
			4: self.tokens[(len(self.tokens)//5)*4: (len(self.tokens)//5)*5],
		}
		all_filtered_words = []
		for zettel in switch.get(thread_index):
			filtered_words = []
			for word in zettel:
				if word not in self.stop_words and word.lower() not in self.stop_words:
					filtered_words.append(str(word))
			all_filtered_words.append(filtered_words)
		self.stop_word_q.put([all_filtered_words, thread_index])

	def pos_tagger(self):
		tokens_with_pos_tags = []
		self.pos_tokens_q = Queue()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			for num in range(5):
				executor.submit(self.get_pos_tags, num)
		results = {}
		while not self.pos_tokens_q.empty():
			temp_result = self.pos_tokens_q.get()
			results[temp_result[1]] = temp_result[0]
		for index in range(5):
			for zettel in results[index]:
				tokens_with_pos_tags.append(zettel)
		return tokens_with_pos_tags

	def get_pos_tags(self, thread_index):
		switch = {
			0: self.filtered_words[:(len(self.filtered_words)//5)],
			1: self.filtered_words[(len(self.filtered_words)//5): (len(self.filtered_words)//5)*2],
			2: self.filtered_words[(len(self.filtered_words)//5)*2: (len(self.filtered_words)//5)*3],
			3: self.filtered_words[(len(self.filtered_words)//5)*3: (len(self.filtered_words)//5)*4],
			4: self.filtered_words[(len(self.filtered_words)//5)*4: (len(self.filtered_words)//5)*5],
		}
		all_tokens_with_pos_tags = []
		for zettel in switch.get(thread_index):
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
		self.pos_tokens_q.put([all_tokens_with_pos_tags, thread_index])

	def create_lemmatized_tokens(self):
		all_lemmatized_tokens = []
		self.lemma_q = Queue()
		wn.ensure_loaded()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			for num in range(5):
				executor.submit(self.get_lemma_tokens, num)
		results = {}
		while not self.lemma_q.empty():
			temp_result = self.lemma_q.get()
			results[temp_result[1]] = temp_result[0]
		for index in range(5):
			for zettel in results[index]:
				all_lemmatized_tokens.append(zettel)
		return all_lemmatized_tokens

	def get_lemma_tokens(self, thread_index):
		switch = {
			0: self.pos_tagged_tokens[:(len(self.pos_tagged_tokens)//5)],
			1: self.pos_tagged_tokens[(len(self.pos_tagged_tokens)//5): (len(self.pos_tagged_tokens)//5)*2],
			2: self.pos_tagged_tokens[(len(self.pos_tagged_tokens)//5)*2: (len(self.pos_tagged_tokens)//5)*3],
			3: self.pos_tagged_tokens[(len(self.pos_tagged_tokens)//5)*3: (len(self.pos_tagged_tokens)//5)*4],
			4: self.pos_tagged_tokens[(len(self.pos_tagged_tokens)//5)*4: (len(self.pos_tagged_tokens)//5)*5],
		}
		all_lemmatized_tokens = []
		for zettel in switch.get(thread_index):
			lemmatized_tokens = []
			for word in zettel:
				lemmatized_tokens.append([self.lemmatizer.lemmatize(word[0], word[2]), word[1]])
			all_lemmatized_tokens.append(lemmatized_tokens)
		self.lemma_q.put([all_lemmatized_tokens, thread_index])

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

	def get_bi_gram(self):
		self.bi_gram = self.create_n_gram(2)

	def get_tri_gram(self):
		self.tri_gram = self.create_n_gram(3)

	def create_unique_corpus(self):
		token_set = []
		for zettel in self.lemmatized_tokens:
			for word in zettel:
				if word not in token_set:
					token_set.append(word[0])
		return token_set

	def get_zettels_from_directory(self, directory):
		new_zettels = []
		files = os.listdir(directory)
		for file in files:
			path = directory + "/" + file
			contents = [str([line.rstrip() for line in open(path)])]
			new_zettels.append(contents)
		return new_zettels

	# def stemmer(self, stemmer_type):
	# 	switch = {
	# 		'porter': PorterStemmer(),
	# 		'lancaster': LancasterStemmer(),
	# 		'snowball': SnowballStemmer('english'),
	# 	}
	# 	stemmer = switch.get(stemmer_type)
	# 	all_stemmed_tokens = []
	# 	for zettel in self.pos_tagged_tokens:
	# 		stemmed_tokens = []
	# 		for word in zettel:
	# 			stemmed_tokens.append([stemmer.stem(word[0]), word[2]])
	# 		all_stemmed_tokens.append(stemmed_tokens)
	# 	return all_stemmed_tokens

	def create_count_matrix(self):  #TODO check
		count_matrix = []
		for zettel in self.zettels:
			count = ZettelPreProcessor.get_word_count(self, zettel)
			count_matrix.append(count)
		return count_matrix

	def get_word_count(self, zettel):  #TODO check
		new_unique_corpus = self.create_unique_corpus()
		count = np.zeros(len(new_unique_corpus))
		split_zettel = re.split("\W+", str(zettel).lower())
		for word in split_zettel:
			new_iter = iter(self.create_unique_corpus())
			i = 0
			for new_word in new_iter:
				if word == new_word:
					count[i] = count[i] + 1
				i += 1
		return count.tolist()

	def get_document_word_counts(self):
		i = 0
		counts = []
		for zettel in self.tokens:
			counts.append(0)
			for word in zettel:
				counts[i] = counts[i] + 1
			i += 1
		return counts

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


if __name__ == "__main__":
	baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
	bibs = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/bibs"
	examples = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/examples"
	rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"

	import datetime
	print(datetime.datetime.now())

	process = ZettelPreProcessor()
	zettels = process.get_zettels_from_directory(baseball)
	process.init_zettels(zettels)

	print("Done.")
	print(datetime.datetime.now())


	# distance = distance.Distance()
	# distance_type = 'euclidean'
	# matrix = np.array(distance.get_distance_matrix(process.unique_count_matrix, distance_type))
	# tf_idf = distance.tf_idf(zettels)
	#
	# cluster = cluster.Cluster()
	# hierarchical_cluster = cluster.hclust(matrix, distance_type)
	# hierarchical_cluster = cluster.hclust(tf_idf, 'tf idf')
	# k_means = cluster.k_means(matrix, distance_type)
