from src import ZettelPreProcessor as PreProcess

import unittest
import numpy as np


class Test(unittest.TestCase):

	sentence1 = "This is a test sentence for data mining zettels zettels zettels."
	sentence2 = "To see the full affect of the functions, this is another test sentence."
	tags = "tags: zettel, sentence, test, cite:..."
	document = [sentence1, sentence2, tags]

	process = PreProcess.ZettelPreProcessor()
	process.init_zettels(document)
	tokens = process.tokenizer()
	unique_corpus = process.create_unique_corpus(tokens)
	count_matrix = process.create_count_matrix(unique_corpus)
	n_gram = process.create_n_gram(tokens, 2)
	filtered_words = process.remove_stop_words(tokens)
	unique_tag_corpus = process.create_unique_tag_corpus(tokens)
	tag_boolean_matrix = process.create_boolean_tag_matrix(unique_tag_corpus)

	distance = []
	example_distances = []
	distance_matrix = []

	def testFunction(self, actual, expected):
		self.assertEqual(actual, expected)

	def test_tokenizer(self):
		expected = ['this',	'is', 'a', 'test', 'sentence', 'for', 'data', 'mining',
					'zettels', 'zettels', 'zettels', '', 'to', 'see', 'the',
					'full', 'affect', 'of', 'the', 'functions', 'this',
					'is', 'another', 'test', 'sentence', '', 'tags', 'zettel',
					'sentence', 'test', 'cite', '']
		self.testFunction(self.tokens, expected)

	def test_unique_corpus(self):
		expected = ["a", "affect", "another", "cite", "data", "for", "full", "functions",
			"is", "mining", "of", "see", "sentence", "tags", "test", "the", "this", 
			"to", "zettel", "zettels"]
		self.testFunction(self.unique_corpus, expected)

	def test_stop_words(self):
		expected = ['test', 'sentence', 'data', 'mining', 'zettels', 'zettels',
					'zettels', '', 'see', 'full', 'affect', 'functions',
					'another', 'test', 'sentence', '', 'tags', 'zettel',
					'sentence', 'test', 'cite', '']
		self.testFunction(self.filtered_words, expected)

	def test_count_matrix(self):
		expected = [[[1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1.]],
					[[1., 0., 1., 1., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0.]],
					[[1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0.]]]
		self.testFunction(self.count_matrix, expected)

	def test_n_gramer(self):
		expected =  [['this', 'is'], ['is', 'a'], ['a', 'test'], ['test', 'sentence'],
					 ['sentence', 'for'], ['for', 'data'], ['data', 'mining'],
					 ['mining', 'zettels'], ['zettels', 'zettels'], ['zettels', 'zettels'],
					 ['zettels', ''], ['', 'to'], ['to', 'see'], ['see', 'the'],
					 ['the', 'full'], ['full', 'affect'], ['affect', 'of'],
					 ['of', 'the'], ['the', 'functions'], ['functions', 'this'],
					 ['this', 'is'], ['is', 'another'], ['another', 'test'],
					 ['test', 'sentence'], ['sentence', ''], ['', 'tags'],
					 ['tags', 'zettel'], ['zettel', 'sentence'], ['sentence', 'test'],
					 ['test', 'cite'],  ['cite', '']]
		self.testFunction(self.n_gram, expected)

	def test_unique_tags(self):
		expected = ["sentence", "tags", "test", "zettel"]
		self.testFunction(self.unique_tag_corpus, expected)

	def test_unique_tag_boolean_matrix(self):
		expected = [[True, True, False], [True, True, False], [True, True, True]]
		self.testFunction(self.tag_boolean_matrix, expected)

	def test_jaccard(self):
		expected = (0.0, 0.4, 0.35)
		self.testFunction(self.distance, expected)

	def test_cosine(self):
		expected = (0.0, 0.2004459314343183, 0.23904572186687872)
		self.testFunction(self.distance, expected)
		
	def test_euclidean(self):
		expected = (0.0, 4.898979485566356, 3.872983346207417)
		self.testFunction(self.distance, expected)
		
	def test_manhattan(self):
		expected = (0.0, 16.0, 13.0)
		self.testFunction(self.distance, expected)

	def test_distance_matrix(self):
		expected = ((1.0, 2.0, 3.0), (3.0, 1.0, 2.0),(2.0, 3.0, 1.0))	
		self.testFunction(self.distance_matrix, expected)


if __name__ == '__main__':
	unittest.main()