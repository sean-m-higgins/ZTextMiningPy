
class Weights:

    def __init__(self):
        self.all_score_weights = [0.3, 0.1, 0.1, 0.25, 0.25]  # [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        self.pos_switch = {  #TODO subject to change
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
        self.z_area_switch = {  #TODO subject to change
            0: 0.80,
            1: 0.60,
            2: 0.40
        }
        self.n_gram_min_freq = 2  #TODO possibly change to 1?