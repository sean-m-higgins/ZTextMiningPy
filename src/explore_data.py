# # TODO histograms of number of tags, num words / num tags...?
from zettel_preprocessor import ZettelPreProcessor
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt

class exploreData:
    def get_wordcloud(self, words, stopwords):
        wordcloud = WordCloud(width = 800, height = 800,
                              background_color ='white',
                              stopwords = stopwords,
                              min_font_size = 10).generate(words)
        return wordcloud

    def plot_wordcloud(self, wordcloud):
        # plot the WordCloud image
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()

    def get_bar_graph(self, values):
        """
        Function to plot bar graph
        :param documents: list of zettels
        """
        n = len(values)
        x = np.arange(n)
        indices = range(0, n)
        plt.bar(x, values, tick_label=indices, align='center')
        plt.xlabel('Zettel #')
        plt.ylabel('Number of Words')
        plt.show()

if __name__ == "__main__":
    baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
    bibs = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/bibs"
    examples = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/examples"
    rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"

    process = ZettelPreProcessor()
    zettels = process.get_zettels_from_directory(baseball)
    process.init_zettels(zettels)

    words = process.lemmatized_tokens
    stopwords = STOPWORDS
    one_str_words = ''

    for word in words:
        one_str_words = one_str_words + word + " "

    ex = exploreData()
    wordcloud = ex.get_wordcloud(one_str_words, stopwords)
    ex.plot_wordcloud(wordcloud)

    document_word_counts = process.get_document_word_counts()
    ex.get_bar_graph(document_word_counts)




'#   #1 Convert source to List of String' \
'#2 Make list of words unique' \
'#3 Make 1&2 a function' \
'#4 Create class to process all zettels' \
'#5 Form unique word corpus' \
'#6 Apply hierarchical clustering methods agglomerative, ... kmeans...??' \
'#7 create matrix of word counts of the files words in uniqueCorpus' \
'#8 tokenize corpus' \
'#9 remove stop words' \
'#10 lemmatize/stem' \
'#11 function to write to text file' \
'TODO #12 LDA' \
'#13 n-grams' \
'#14 word,count dictionary' \
'#15 visual graphs' \
'#16 test Suite' \
'#17 distances - euclidean, manhattan, cosine, jaccard, minkowski, tf_idf...bootstrap?...spearmans_rank?...??' \
'#18 part of speech' \
'#19 uniqueTagCorpus' \
'#21 normalize tf_idf' \
'TODO #22 retrieval class' \
'TODO #23 multi-label classification' \
'TODO #24 word embeddings' \
'TODO #25 '