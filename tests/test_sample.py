from src import ZettelPreProcessor as PreProcess
from src import Distance

import unittest


class Test(unittest.TestCase):

	sentence1 = "This is a test sentence for data mining zettels zettels zettels."
	sentence2 = "To see the full affect of the functions, this is another test sentence."
	tags = "tags: zettel, sentence, test, cite:..." #TODO change all back
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

	distance = Distance.Distance()
	example_distances = [1.0, 2.0, 3.0, 4.0]

	def testFunction(self, actual, expected):
		self.assertEqual(actual, expected)

	def test_tokenizer(self):
		expected = ['this',	'is', 'a', 'test', 'sentence', 'for', 'data', 'mining', 'zettels', 'zettels', 'zettels',
					'to', 'see', 'the', 'full', 'affect', 'of', 'the', 'functions', 'this', 'is', 'another', 'test',
					'sentence', 'tags', 'zettel', 'sentence', 'test', 'cite']
		self.testFunction(self.tokens, expected)

	def test_unique_corpus(self):
		expected = ["a", "affect", "another", "cite", "data", "for", "full", "functions", "is", "mining", "of", "see",
					"sentence", "tags", "test", "the", "this", "to", "zettel", "zettels"]
		self.testFunction(self.unique_corpus, expected)

	def test_stop_words(self):
		expected = ['test', 'sentence', 'data', 'mining', 'zettels', 'zettels', 'zettels', 'see', 'full', 'affect',
					'functions', 'another', 'test', 'sentence', 'tags', 'zettel', 'sentence', 'test', 'cite']
		self.testFunction(self.filtered_words, expected)

	def test_count_matrix(self):
		expected = [[1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1.,0., 0., 1.],
				[0., 1., 1., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.,1., 0., 0.],
				[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0.]]
		self.testFunction(self.count_matrix, expected)

	def test_n_gramer(self):
		expected =  [['this', 'is'], ['is', 'a'], ['a', 'test'], ['test', 'sentence'], ['sentence', 'for'],
					  ['for', 'data'], ['data', 'mining'], ['mining', 'zettels'], ['zettels', 'zettels'],
					  ['zettels', 'zettels'], ['zettels', 'to'], ['to', 'see'], ['see', 'the'], ['the', 'full'],
					  ['full', 'affect'], ['affect', 'of'], ['of', 'the'], ['the', 'functions'], ['functions', 'this'],
					  ['this', 'is'],  ['is', 'another'], ['another', 'test'], ['test', 'sentence'], ['sentence', 'tags'],
					  ['tags', 'zettel'], ['zettel', 'sentence'], ['sentence', 'test'], ['test', 'cite']]
		self.testFunction(self.n_gram, expected)

	def test_unique_tags(self):
		expected = ["sentence", "test", "zettel"]
		self.testFunction(self.unique_tag_corpus, expected)

	def test_unique_tag_boolean_matrix(self):
		expected = [[True, True, False], [True, True, False], [True, True, True]]
		self.testFunction(self.tag_boolean_matrix, expected)

	def test_euclidean(self):
		expected = [0, 3.605551275463989, 3.605551275463989]
		distance = self.distance.calculate_distances(self.count_matrix, 0)
		self.testFunction(distance, expected)

	def test_manhattan(self):
		expected = [0, 13.0, 13.0]
		distance = self.distance.calculate_distances(self.count_matrix, 1)
		self.testFunction(distance, expected)

	def test_minkowsky(self):
		expected = [0, 2.3513346877207573, 2.3513346877207573]
		distance = self.distance.calculate_distances(self.count_matrix, 2)
		self.testFunction(distance, expected)

	def test_cosine(self):
		expected = [0, 0.3849001794597505, 0.2581988897471611]
		distance = self.distance.calculate_distances(self.count_matrix, 3)
		self.testFunction(distance, expected)

	def test_jaccard(self):
		expected = [0, 0.05263157894736842, 0.05263157894736842]
		distance = self.distance.calculate_distances(self.count_matrix, 4)
		self.testFunction(distance, expected)

	def test_distance_matrix(self):
		expected = [[1.0, 2.0, 3.0, 4.0],
					[4.0, 1.0, 2.0, 3.0],
					[3.0, 4.0, 1.0, 2.0],
					[2.0, 3.0, 4.0, 1.0]]
		distance_matrix = self.distance.create_distance_matrix(self.example_distances)
		self.testFunction(distance_matrix, expected)


if __name__ == '__main__':
	unittest.main()