
class Weights:

    def __init__(self):
        # tf_idf @ 0; word_score @ 1; keyword_score @ 2; text_rank @ 3; pos_score @ 4; area_score @ 5
        self.all_score_weights = [0.50, 0.05, 0.20, 0.05, 0.1, 0.1]  #[0.40, 0.05, 0.30, 0.05, 0.1, 0.1]  #[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        self.pos_switch = {
            'NN': 0.40,
            'NNS': 0.35,
            'NNP': 0.80,
            'NNPS': 0.70,
            'NG': 0.50,
            'VB': 0.25,
            'VBD': 0.25,
            'VBG': 0.25,
            'VBN': 0.25,
            'VBP': 0.25,
            'VBZ': 0.25,
            'JJ': 0.15,
            'JJR': 0.15,
            'JJS': 0.15,
            'RB': 0.15,
            'RBR': 0.15,
            'RBS': 0.15
        }
        self.z_area_switch = {
            0: 0.80,
            1: 0.60,
            2: 0.40
        }
        self.n_gram_min_freq = 2
        self.keyword_n = 7