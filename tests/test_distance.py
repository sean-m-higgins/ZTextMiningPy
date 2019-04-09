from src import ZettelPreProcessor, Distance
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
    count_dict = process.create_count_dictionary(tokens)
    doc_count_dict = process.create_doc_count_dictionary(tokens)

    distance = Distance.Distance()
    example_distances = [1.0, 2.0, 3.0, 4.0]

    def test_euclidean(self):
        expected = [0, 4.898979485566356, 4.0]
        distance = self.distance.calculate_distances(self.count_matrix, 0)
        self.assertEqual(distance, expected)

    def test_manhattan(self):
        expected = [0, 16.0, 14.0]
        distance = self.distance.calculate_distances(self.count_matrix, 1)
        self.assertEqual(distance, expected)

    def test_minkowsky(self):
        expected = [0, 3.583047871015946, 2.7144176165949063]
        distance = self.distance.calculate_distances(self.count_matrix, 2)
        self.assertEqual(distance, expected)

    def test_cosine(self):
        expected = [0, 0.25048971643405976, 0.23094010767585027]
        distance = self.distance.calculate_distances(self.count_matrix, 3)
        self.assertEqual(distance, expected)

    def test_jaccard(self):
        expected = [0, 0.05263157894736842, 0.05263157894736842]
        distance = self.distance.calculate_distances(self.count_matrix, 4)
        self.assertEqual(distance, expected)

    def test_df_idf(self):
        expected = [0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896,
                    0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896,
                    0.3103448275862069, 0.3103448275862069, 0.3103448275862069, 0.10344827586206896,
                    0.10344827586206896, 0.20689655172413793, 0.10344827586206896, 0.10344827586206896,
                    0.10344827586206896, 0.20689655172413793, 0.10344827586206896, 0.10344827586206896,
                    0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896,
                    0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896,
                    0.10344827586206896]
        distance = self.distance.tf_idf(self.document)
        self.assertEqual(distance, expected)

    def test_distance_matrix(self):
        expected = [[1.0, 2.0, 3.0, 4.0],
                    [4.0, 1.0, 2.0, 3.0],
                    [3.0, 4.0, 1.0, 2.0],
                    [2.0, 3.0, 4.0, 1.0]]
        distance_matrix = self.distance.create_distance_matrix(self.example_distances)
        self.assertEqual(distance_matrix, expected)