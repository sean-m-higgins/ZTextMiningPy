import unittest

class Test(unittest.TestCase):

	sentence1 = "This is a test sentence for data mining zettels zettels zettels."
	sentence2 = "To see the full affect of the functions, this is another test sentence."
	tags = "tags: zettel, sentence, test, cite:..."
	document = (sentence1, sentence2, tags)

	preProcess
	singleTokens
	uniqueCorpus
	stopWords
	nGram
	countMatrix
	uniqueTags
	uniqueTagBooleanMatrix

	distance
	exampleDistances
	distanceMatrix


	def testFunction(actual, expected):
		assertEqual(actual,expected)


	def test_tokenizer:
		expected = ("this", "is", "a", "test", "sentence", "for", "data", "mining", 
			"zettels", "zettels", "zettels")
		testFunction(singleTokens, expected)

	def test_unique_corpus:
		expected = ("a", "affect", "another", "cite", "data", "for", "full", "functions",
			"is", "mining", "of", "see", "sentence", "tags", "test", "the", "this", 
			"to", "zettel", "zettels")
		testFunction(uniqueCorpus, expected)

	def test_stop_words:
		expected = ("test", "sentence", "data", "mining", "zettels", "zettels", "zettels")
		testFunction(stopWords, expected)

	def test_count_matrix:
		expected = ((1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 
			0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0),
        	(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 
        		2.0, 1.0, 0.0, 0.0, 0.0),
        	(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 
        		0.0, 0.0, 0.0, 1.0, 0.0))
		testFunction(countMatrix, expected)

	def test_n_gramer:
		expected = ("test sentence", "sentence data", "data mining", "mining zettels", 
			"zettels zettels", "zettels zettels")
		testFunction(nGram, expected)

	def test_unique_tags:
		expected = ("sentence", "tags", "test", "zettel")
		testFunction(uniqueTags, expected)

	def test_unique_tag_boolean_matrix:
		expected = ((true, false, true, false), (true, false, true, false),
      (true, true, true, true))
		testFunction(uniqueTagBooleanMatrix, expected)

	def test_jaccard:
		expected = (0.0, 0.4, 0.35)
		testFunction(distances, expected)

	def test_cosine:
		expected = (0.0, 0.2004459314343183, 0.23904572186687872)
		testFunction(distances, expected)
		
	def test_euclidean:
		expected = (0.0, 4.898979485566356, 3.872983346207417)
		testFunction(distances, expected)
		
	def test_manhattan:
		expected = (0.0, 16.0, 13.0)
		testFunction(distances, expected)	

	def test_distance_matrix:
		expected = ((1.0, 2.0, 3.0), (3.0, 1.0, 2.0),(2.0, 3.0, 1.0))	
		testFunction(distanceMatrix, expected)	


if _name_ == '_main_':
	unittest.main()