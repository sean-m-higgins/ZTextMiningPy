from src import ZettelPreProcessor

import unittest


class Test(unittest.TestCase):

    sentence1 = "This is a test sentence for data mining zettels zettels zettels."
    sentence2 = "To see the full affect of the functions, this is another test sentence."
    tags = "tags: zettel, sentence, test, cite:..."
    document = [sentence1, sentence2, tags]

    process = ZettelPreProcessor.ZettelPreProcessor()
    process.init_zettels(document)
    tokens = process.tokenizer()
    unique_corpus = process.create_unique_corpus(tokens)
    count_matrix = process.create_count_matrix(unique_corpus)
    n_gram = process.create_n_gram(tokens, 2)
    filtered_words = process.remove_stop_words(tokens)
    unique_tag_corpus = process.create_unique_tag_corpus(tokens)
    tag_boolean_matrix = process.create_boolean_tag_matrix(unique_tag_corpus)
    pos_tokens = process.pos_tagger(tokens)
    lemmatized = process.lematizer(pos_tokens)
    stemmed = process.stemmer(tokens, 'lancaster')
    count_dict = process.create_count_dictionary(tokens)
    doc_count_dict = process.create_doc_count_dictionary(tokens)

    def test_tokenizer(self):
        expected = ['this',	'is', 'a', 'test', 'sentence', 'for', 'data', 'mining', 'zettels', 'zettels', 'zettels',
                    'to', 'see', 'the', 'full', 'affect', 'of', 'the', 'functions', 'this', 'is', 'another', 'test',
                    'sentence', 'tags', 'zettel', 'sentence', 'test', 'cite']
        self.assertEqual(self.tokens, expected)

    def test_unique_corpus(self):
        expected = ["a", "affect", "another", "cite", "data", "for", "full", "functions", "is", "mining", "of", "see",
                    "sentence", "tags", "test", "the", "this", "to", "zettel", "zettels"]
        self.assertEqual(self.unique_corpus, expected)

    def test_stop_words(self):
        expected = ['test', 'sentence', 'data', 'mining', 'zettels', 'zettels', 'zettels', 'see', 'full', 'affect',
                    'functions', 'another', 'test', 'sentence', 'tags', 'zettel', 'sentence', 'test', 'cite']
        self.assertEqual(self.filtered_words, expected)

    def test_count_matrix(self):
        expected = [[1.0,  0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
        self.assertEqual(self.count_matrix, expected)

    def test_n_gramer(self):
        expected =  [['this', 'is'], ['is', 'a'], ['a', 'test'], ['test', 'sentence'], ['sentence', 'for'],
                     ['for', 'data'], ['data', 'mining'], ['mining', 'zettels'], ['zettels', 'zettels'],
                     ['zettels', 'zettels'], ['zettels', 'to'], ['to', 'see'], ['see', 'the'], ['the', 'full'],
                     ['full', 'affect'], ['affect', 'of'], ['of', 'the'], ['the', 'functions'], ['functions', 'this'],
                     ['this', 'is'],  ['is', 'another'], ['another', 'test'], ['test', 'sentence'], ['sentence', 'tags'],
                     ['tags', 'zettel'], ['zettel', 'sentence'], ['sentence', 'test'], ['test', 'cite']]
        self.assertEqual(self.n_gram, expected)

    def test_unique_tags(self):
        expected = ["sentence", "test", "zettel"]
        self.assertEqual(self.unique_tag_corpus, expected)

    def test_unique_tag_boolean_matrix(self):
        expected = [[True, True, False], [True, True, False], [True, True, True]]
        self.assertEqual(self.tag_boolean_matrix, expected)

    def test_lemmatizer(self):
        expected = ['be', 'test', 'sentence', 'data', 'mining', 'zettels', 'zettels', 'zettels', 'see', 'full',
                    'affect', 'function', 'be', 'test', 'sentence', 'tag', 'zettel', 'sentence', 'test', 'cite']
        self.assertEqual(self.lemmatized, expected)

    def test_pos_tagger(self):
        expected = [['is', 'v'], ['test', 'n'], ['sentence', 'n'], ['data', 'n'], ['mining', 'n'], ['zettels', 'n'],
                    ['zettels', 'v'], ['zettels', 'n'], ['see', 'v'], ['full', 'a'], ['affect', 'n'],
                    ['functions', 'n'], ['is', 'v'], ['test', 'n'], ['sentence', 'n'], ['tags', 'n'], ['zettel', 'n'],
                    ['sentence', 'n'], ['test', 'n'], ['cite', 'n']]
        self.assertEqual(self.pos_tokens, expected)

    def test_stemmer(self):
        expected = ['thi', 'is', 'a', 'test', 'sent', 'for', 'dat', 'min', 'zettel', 'zettel', 'zettel', 'to', 'see',
                    'the', 'ful', 'affect', 'of', 'the', 'funct', 'thi', 'is', 'anoth', 'test', 'sent', 'tag', 'zettel',
                    'sent', 'test', 'cit']
        self.assertEqual(self.stemmed, expected)

    def test_count_dict(self):
        expected = {'a': 1, 'affect': 1, 'another': 1, 'cite': 1, 'data': 1, 'for': 1, 'full': 1, 'functions': 1,
                    'is': 2, 'mining': 1, 'of': 1, 'see': 1, 'sentence': 3, 'tags': 1, 'test': 3, 'the': 2, 'this': 2,
                    'to': 1, 'zettel': 1, 'zettels': 3}
        self.assertEqual(self.count_dict, expected)

    def test_doc_count_dict(self):
        expected = {'a': 1, 'affect': 1, 'another': 1, 'cite': 1, 'data': 1, 'for': 1, 'full': 1, 'functions': 1,
                    'is': 2, 'mining': 1, 'of': 1, 'see': 1, 'sentence': 3, 'tags': 1, 'test': 3, 'the': 1, 'this': 2,
                    'to': 1, 'zettel': 1, 'zettels': 1}
        self.assertEqual(self.doc_count_dict, expected)


if __name__ == '__main__':
    unittest.main()