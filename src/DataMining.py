from src import ZettelPreProcessor as PreProcess


class DataMining:

    zettels = ["THis is a test", "TEST, test...zettel", "tags: test, sentence, zettel, cite"]

    process = PreProcess.ZettelPreProcessor()
    process.init_zettels(zettels)
    tokens = process.tokenizer()
    unique_corpus = process.create_unique_corpus(tokens)
    count_matrix = process.create_count_matrix(unique_corpus)
    n_gram = process.create_n_gram(tokens, 2)
    stop_words = process.remove_stop_words(tokens)
    unique_tag_corpus = process.create_unique_tag_corpus(tokens)
    tag_boolean_matrix = process.create_boolean_tag_matrix(unique_tag_corpus)

    print(tokens)
    print(unique_corpus)
    print(unique_tag_corpus)
    print(count_matrix)
    print(tag_boolean_matrix)
    print(n_gram)
    print(stop_words)


