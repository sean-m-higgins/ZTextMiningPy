# Keyword extraction using the RAKE library... https://github.com/fabianvf/python-rake
import RAKE
import zettel_preprocessor as process

stopwords = "/Users/SeanHiggins/ZTextMiningPy/docs/data/processedData/stopWords/zettelStopWords.txt"
clean_baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"

z_process = process.ZettelPreProcessor()
zettels = z_process.get_zettels_from_clean_directory(clean_baseball)
rake = RAKE.Rake(stopwords, regex='\W+')

for zettel in zettels:
    content = ""
    for section in zettel:
        content = content + section + " "
    output = rake.run(content, minCharacters=1, maxWords=3, minFrequency=2)
    print(output)
    print("\n")
