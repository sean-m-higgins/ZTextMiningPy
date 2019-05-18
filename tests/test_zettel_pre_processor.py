from src import zettel_preprocessor
import unittest


class Test(unittest.TestCase):

    sentence1 = "This is a test sentence for data mining zettels zettels zettels."
    sentence2 = "To see the full affect of the functions, this is another test sentence."
    tags = "tags: zettel, sentence, test, cite:..."
    document = [[sentence1], [sentence2], [tags]]


    process = zettel_preprocessor.ZettelPreProcessor()
    process.init_zettels(document)
    tokens = process.tokenizer()
    pos_tokens = process.pos_tagger()
    filtered_words = process.remove_stop_words()
    # stemmer types: 'porter', 'lancaster', 'snowball'
    stemmed_tokens = process.stemmer('lancaster')
    lemmatized_tokens = process.lemmatizer()
    n_grams = process.create_n_gram(2)
    unique_corpus = process.create_unique_corpus()
    unique_n_gram_corpus = process.create_unique_corpus()
    #unique_tag_corpus = process.create_unique_tag_corpus(tags)  #TODO
    #tag_boolean_matrix = process.create_boolean_tag_matrix(unique_tag_corpus) #TODO
    count_matrix = process.create_count_matrix()
    count_dict = process.create_count_dictionary()
    doc_count_dict = process.create_doc_count_dictionary()


    def test_tokenizer(self):
        expected = ['This',	'is', 'a', 'test', 'sentence', 'for', 'data', 'mining', 'zettels', 'zettels', 'zettels',
                    'To', 'see', 'the', 'full', 'affect', 'of', 'the', 'functions', 'this', 'is', 'another', 'test',
                    'sentence', 'tags', 'zettel', 'sentence', 'test', 'cite']
        self.assertEqual(self.tokens, expected)

    def test_unique_corpus(self):
        expected = ['affect', 'cit', 'dat', 'ful', 'funct', 'min', 'see', 'sent', 'tag', 'test', 'zettel']
        self.assertEqual(self.unique_corpus, expected)

    def test_stop_words(self):
        expected = [['test', 'n'], ['sentence', 'n'], ['data', 'n'], ['mining', 'n'], ['zettels', 'n'],
                    ['zettels', 'v'], ['zettels', 'n'], ['see', 'v'], ['full', 'a'], ['affect', 'n'],
                    ['functions', 'n'], ['test', 'n'], ['sentence', 'n'], ['tags', 'n'], ['zettel', 'n'],
                    ['sentence', 'n'], ['test', 'n'], ['cite', 'n']]
        self.assertEqual(self.filtered_words, expected)

    def test_count_matrix(self):
        expected = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]
        self.assertEqual(self.count_matrix, expected)

    def test_n_gramer(self):
        expected = ['test sent', 'sent dat', 'dat min', 'min zettel', 'zettel zettel', 'zettel zettel', 'zettel see',
                    'see ful', 'ful affect', 'affect funct', 'funct test', 'test sent', 'sent tag', 'tag zettel',
                    'zettel sent', 'sent test',  'test cit']
        self.assertEqual(self.n_grams, expected)

    def test_lemmatizer(self):
        expected = ['test', 'sent', 'dat', 'min', 'zettel', 'zettel', 'zettel', 'see', 'ful', 'affect', 'funct', 'test',
                    'sent', 'tag', 'zettel', 'sent', 'test', 'cit']
        self.assertEqual(self.lemmatized_tokens, expected)

    def test_pos_tagger(self):
        expected = [['is', 'v'], ['test', 'n'], ['sentence', 'n'], ['data', 'n'], ['mining', 'n'], ['zettels', 'n'],
                    ['zettels', 'v'], ['zettels', 'n'], ['see', 'v'], ['full', 'a'], ['affect', 'n'],
                    ['functions', 'n'], ['is', 'v'], ['test', 'n'], ['sentence', 'n'], ['tags', 'n'], ['zettel', 'n'],
                    ['sentence', 'n'], ['test', 'n'], ['cite', 'n']]
        self.assertEqual(self.pos_tokens, expected)

    def test_stemmer(self):
        expected = [['test', 'n'], ['sent', 'n'], ['dat', 'n'], ['min', 'n'], ['zettel', 'n'], ['zettel', 'v'],
                    ['zettel', 'n'], ['see', 'v'], ['ful', 'a'], ['affect', 'n'], ['funct', 'n'], ['test', 'n'],
                    ['sent', 'n'], ['tag', 'n'], ['zettel', 'n'], ['sent', 'n'], ['test', 'n'], ['cit', 'n']]
        self.assertEqual(self.stemmed_tokens, expected)

    def test_count_dict(self):
        expected = {'affect': 1, 'cit': 1, 'dat': 1, 'ful': 1, 'funct': 1, 'min': 1, 'see': 1, 'sent': 3, 'tag': 1,
                    'test': 3, 'zettel': 4}
        self.assertEqual(self.count_dict, expected)

    def test_doc_count_dict(self):
        expected = {'affect': 1, 'cit': 1, 'dat': 1, 'ful': 1, 'funct': 1, 'min': 1, 'see': 1, 'sent': 3, 'tag': 1,
                    'test': 3, 'zettel': 2}
        self.assertEqual(self.doc_count_dict, expected)

    # def test_unique_tags(self):
    #     expected = ["sentence", "test", "zettel"]
    #     self.assertEqual(self.unique_tag_corpus, expected)

    # def test_unique_tag_boolean_matrix(self):
    #     expected = [[True, True, False], [True, True, False], [True, True, True]]
    #     self.assertEqual(self.tag_boolean_matrix, expected)


if __name__ == '__main__':
    Test.init_tests()
    unittest.main()
