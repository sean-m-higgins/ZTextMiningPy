import queue
import zettel_preprocess
import zettel_ke

q = queue.Queue()

zettel = ["Casablanca",
        "A cynical American expatriate struggles to decide whether or not he should help his former lover and her fugitive husband escape French Morocco.",
        "The story of Rick Blaine, a cynical world-weary ex-patriate who runs a nightclub in Casablanca, Morocco during the early stages of WWII. Despite the pressure he constantly receives from the local authorities, Rick's cafe has become a kind of haven for refugees seeking to obtain illicit letters that will help them escape to America. But when Ilsa, a former lover of Rick's, and her husband, show up to his cafe one day, Rick faces a tough challenge which will bring up unforeseen complications, heartbreak and ultimately an excruciating decision to make. During World War II, Europeans who were fleeing from the Germans, sought refuge in America. But to get there they would first have to go Casablanca and once they get there, they have to obtain exit visas which are not very easy to come by. Now the hottest spot in all of Casablanca is Rick's Cafe which is operated by Rick Blaine, an American expatriate, who for some reason can't return there, and he is also extremely cynical. Now it seems that two German couriers were killed and the documents they were carrying were taken. Now one of Rick's regulars, Ugarte entrusts to him some letters of transit, which he intends to sell but before he does he is arrested for killing the couriers. Captain Renault, the Chief of Police, who is neutral in his political views, informs Rick that Victor Laszlo, a resistance leader from Czechoslovakia, is in Casablanca and will do anything to get an exit visa but Renault has been told by Major Strasser of the Gestapo, to keep Laszlo in Casablanca. Laszlo goes to Rick's to meet Ugarte, because he was the one Ugarte was going to sell the letters to. But since Ugarte was arrested he has to find another way. Accompanying him is Ilsa Lund, who knew Rick when he was in Paris, and when they meet some of Rick's old wounds reopen. It is obvious that Rick's stone heart was because of her leaving him. And when they learn that Rick has the letters, he refuses to give them to him, because he doesn't stick his neck out for anyone.",
        "nazi;anti nazi;casablanca morocco;french morocco;1940s;love triangle;police"]

q.put(zettel)

tokens = []
if_run = False
while True:
    while not q.empty():
        z_preprocess = zettel_preprocess.ZettelPreProcessor(q.get())
        tokens.append(z_preprocess.pos_tagged_tokens)
        if_run = True
    if if_run:
        z_ke = zettel_ke.ZettelKE(tokens)
        suggested_keywords = z_ke.run()
    if_run = False

index = 0
for zettel in suggested_keywords:
    print("\nSuggested Keywords for Zettel " + str(index) + ": ")
    inner_i = 1
    for item in zettel:
        print(str(inner_i) + ": ")
        print(item)
        inner_i += 1
    index += 1
