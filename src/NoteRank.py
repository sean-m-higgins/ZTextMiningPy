import numpy as np
import zettel_preprocessor as process

class NoteRank(object):
    def __init__(self, zettels):
        z_process = process.ZettelPreProcessor()
        z_process.init_zettels(zettels)

    def run(self):
        return





if __name__ == "__main__":
    rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"
    baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
    movies = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/movies"
    clean_baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"

    import datetime

    print(datetime.datetime.now())

    z_process = process.ZettelPreProcessor()
    zettels = z_process.get_zettels_from_clean_directory(clean_baseball)

    nr = NoteRank(zettels)
    suggested_keywords = nr.run()

    index = 0
    for zettel in suggested_keywords:
        print("\nSuggested Keywords for Zettel " + str(index) + ": ")
        inner_i = 1
        for item in zettel:
            print(str(inner_i) + ": ")
            print(item)
            inner_i += 1
        index += 1

    print("Done.")
    print(datetime.datetime.now())
