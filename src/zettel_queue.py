import zettel_preprocess
import zettel_ke
import queue
import os  #TODO remove

def get_zettels_from_clean_directory(directory):  #TODO remove once connected with DB
    new_zettels = []
    files = os.listdir(directory)
    for file in files:
        path = directory + "/" + file
        zettel = []
        lines = open(path).readlines()
        for line in lines:
            zettel.append(line)
        new_zettels.append(zettel)
    return new_zettels


if __name__ == "__main__":
    movies = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/movies"  #TODO remove
    clean_baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/clean_baseball"  #TODO remove

    import datetime  #TODO remove
    print(datetime.datetime.now())  #TODO remove

    q = queue.Queue()

    zettels = get_zettels_from_clean_directory(movies)  # TODO remove
    for zettel in zettels:  #TODO 
        q.put(zettel)

    zettels2 = get_zettels_from_clean_directory(clean_baseball)  #TODO remove
    for zettel in zettels2:  #TODO 
        q.put(zettel)

    # q.put(["12 Angry Men", 
    #     "The defense and the prosecution have rested and the jury is filing into the jury room to decide if a young man is guilty or innocent of murdering his father.",
    #     "What begins as an open-and-shut case of murder soon becomes a detective story that presents a succession of clues creating doubt, and a mini-drama of each of the jurors' prejudices and preconceptions about the trial, the accused, and each other. Based on the play, all of the action takes place on the stage of the jury room. 12 Angry Men focuses on a jury's deliberations in a capital murder case. A 12-man jury is sent to begin deliberations in the first-degree murder trial of an 18-year-old man accused in the stabbing death of his father, where a guilty verdict means an automatic death sentence. The case appears to be open-and-shut: The defendant has a weak alibi; a knife he claimed to have lost is found at the murder scene; and several witnesses either heard screaming, saw the killing or the boy fleeing the scene. Eleven of the jurors immediately vote guilty; only Juror No. 8 (Mr. Davis) casts a not guilty vote. At first Mr. Davis bases his vote more so for the sake of discussion; after all, the jurors must believe beyond a reasonable doubt that the defendant is guilty. As the deliberations unfold, the story quickly becomes a study of the jurors' complex personalities (which range from wise, bright and empathetic to arrogant, prejudiced and merciless), preconceptions, backgrounds and interactions. That provides the backdrop to Mr. Davis' attempts in convincing the other jurors that a not guilty verdict might be appropriate.", 
    #     "jury;dialogue driven;courtroom;single set production;trial;justice;judicial system"])

    tokens = []  #TODO remove 
    count_dicts = []  #TODO remove 
    given_tags = []  #TODO remove 
    if_run = False
    while True:
        while not q.empty():
            z_preprocess = zettel_preprocess.ZettelPreProcessor(q.get())
            tokens.append(z_preprocess.pos_tagged_tokens)  #TODO add to ML table
            count_dicts.append(z_preprocess.count_dict)  #TODO add to ML table
            given_tags.append(z_preprocess.given_tags)  #TODO add to ML table
            if_run = True
        if if_run:
            if_run = False
            #TODO get all data from ML table (tokens, count_dicts, given_tags)
            z_ke = zettel_ke.ZettelKE(tokens, count_dicts, given_tags)
            suggested_keywords = z_ke.run()
            #TODO send suggested keywords for the bottom row (latest zettel) to appropriate place in table
            index = 0  #TODO remove
            for zettel in suggested_keywords:  #TODO remove
                print("\nSuggested Keywords for Zettel " + str(index) + ": ")
                inner_i = 1
                for item in zettel:
                    print(str(inner_i) + ": ")
                    print(item)
                    inner_i += 1
                index += 1
            print(datetime.datetime.now())  #TODO remove


#TODO-later clean removals
#TODO-later get_keywords()
#TODO-later tokenizer()
#TODO-later ML Table
#TODO get_keyword_score()??
