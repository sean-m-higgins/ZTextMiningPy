# Keyword extraction using the RAKE library... https://github.com/fabianvf/python-rake
import RAKE

rake = RAKE.Rake("/Users/SeanHiggins/ZTextMiningPy/docs/data/processedData/stopWords/zettelStopWords.txt", regex ='[\W\n]+')

output = rake.run("Chapter One: The Computer Revolution Hasn't Happened Yet Chapter Two: The First Programmer Was a "
                  "Lady Chapter Three: The First Hacker and his Imaginary Machine Chapter Four: Johnny Builds Bombs and "
                  "Johnny Builds Brains Chapter Five: Ex-Prodigies and Antiaircraft Guns Chapter Six: Inside Information"
                  " Chapter Seven: Machines to Think With Chapter Eight: Witness to History: The Mascot of Project Mac "
                  "Chapter Nine: The Loneliness of a Long-Distance Thinker Chapter Ten: The New Old Boys from the ARPAnet"
                  " Chapter Eleven: The Birth of the Fantasy Amplifier Chapter Twelve: Brenda and the Future Squad Chapter "
                  "Thirteen: Knowledge Engineers and Epistemological Entrepreneurs Chapter Fourteen: Xanadu, Network "
                  "Culture, and Beyond Chapter One: The Computer Revolution Hasn't Happened Yet South of San Francisco "
                  "and north of Silicon Valley, near the place where the pines on the horizon give way to the live oaks "
                  "and radiotelescopes, an unlikely subculture has been creating a new medium for human thought. When "
                  "mass-production models of present prototypes reach our homes, offices, and schools, our lives are "
                  "going to change dramatically.The first of these mind-amplifying machines will be descendants of the "
                  "devices now known as personal computers, but they will resemble today's information processing "
                  "technology no more than a television resembles a fifteenth-century printing press. They aren't "
                  "available yet, but they will be here soon. Before today's first-graders graduate from high school, "
                  "hundreds of millions of people around the world will join together to create new kinds of human "
                  "communities, making use of a tool that a small number of thinkers and tinkerers dreamed into being "
                  "over the past century. Nobody knows whether this will turn out to be the best or the worst thing the "
                  "human race has done for itself, because the outcome of this empowerment will depend in large part "
                  "on how we react to it and what we choose to do with it. The human mind is not going to be replaced "
                  "by a machine, at least not in the foreseeable future, but there is little doubt that the worldwide "
                  "availability of fantasy amplifiers, intellectual toolkits, and interactive electronic communities "
                  "will change the way people think, learn, and communicate. It looks as if this latest "
                  "technology-triggered transformation of society could have even more intense impact than the last "
                  "time human thought was augmented, five hundred years ago, when the Western world learned to read. "
                  "Less than a century after the invention of movable type, the literate community in Europe had grown "
                  "from a privileged minority to a substantial portion of the population. People's lives changed "
                  "radically and rapidly, not because of printing machinery, but because of what that invention made "
                  "it possible for people to know. Books were just the vehicles by which the ideas escaped from the "
                  "private libraries of the elite and circulated among the population.",
                  minCharacters = 1, maxWords = 3, minFrequency = 2)

print(output)