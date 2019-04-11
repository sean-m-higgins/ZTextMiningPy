from src import zettel_preprocessor, distance
import unittest


class Test(unittest.TestCase):

    sentence1 = "This is a test sentence for data mining zettels zettels zettels."
    sentence2 = "To see the full affect of the functions, this is another test sentence."
    tags = "tags: zettel, sentence, test, cite:..."
    document = [[sentence1], [sentence2], [tags]]

    process = zettel_preprocessor.ZettelPreProcessor()
    process.init_zettels(document)

    tokens = process.tokenizer()
    filtered_words = process.remove_stop_words(tokens)
    pos_tokens = process.pos_tagger(filtered_words)
    # stemmer types: 'porter', 'lancaster', 'snowball'
    stemmed_tokens = process.stemmer(pos_tokens, 'lancaster')
    lemmatized_tokens = process.lematizer(stemmed_tokens)

    n_grams = process.create_n_gram(lemmatized_tokens, 2)

    unique_corpus = process.create_unique_corpus(lemmatized_tokens)

    count_matrix = process.create_count_matrix(unique_corpus)

    count_dict = process.create_count_dictionary(lemmatized_tokens)
    doc_count_dict = process.create_doc_count_dictionary(lemmatized_tokens)

    distance = distance.Distance()

    def test_euclidean(self):
        expected = [0, 1.4142135623730951, 1.7320508075688772]
        distances = self.distance.calculate_distances(self.count_matrix, 0)
        self.assertEqual(distances, expected)

    def test_manhattan(self):
        expected = [0, 2.0, 3.0]
        distances = self.distance.calculate_distances(self.count_matrix, 1)
        self.assertEqual(distances, expected)

    def test_minkowsky(self):
        expected = [0, 1.2599210498948732, 1.4422495703074083]
        distances = self.distance.calculate_distances(self.count_matrix, 2)
        self.assertEqual(distances, expected)

    def test_cosine(self):
        expected = [0, 0.5773502691896258, 0.40824829046386296]
        distances = self.distance.calculate_distances(self.count_matrix, 3)
        self.assertEqual(distances, expected)

    def test_jaccard(self):
        expected = [0, 0.1, 0.1]
        distances = self.distance.calculate_distances(self.count_matrix, 4)
        self.assertEqual(distances, expected)

    def test_df_idf(self):
        expected = [[0.14285714285714285, 0.14285714285714285, 0.42857142857142855, 0.42857142857142855,
                     0.6428571428571428, 0.6428571428571428, 0.6428571428571428],
                    [0.5, 0.5, 0.5, 0.5, 0.16666666666666666, 0.16666666666666666],
                    [0.6000000000000001, 0.30000000000000004, 0.2, 0.2, 0.6000000000000001]]
        distances = self.distance.tf_idf(self.document)
        self.assertEqual(distances, expected)

    def test_distance_matrix(self):
        expected = [[1.0, 2.0, 3.0, 4.0],
                    [4.0, 1.0, 2.0, 3.0],
                    [3.0, 4.0, 1.0, 2.0],
                    [2.0, 3.0, 4.0, 1.0]]
        example_distances = [1.0, 2.0, 3.0, 4.0]
        distance_matrix = self.distance.create_distance_matrix(example_distances)
        self.assertEqual(distance_matrix, expected)
