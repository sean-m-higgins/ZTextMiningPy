class Weights:

    def __init__(self):
        # tf_idf @ 0; keyword_score @ 1; pos_score @ 2; area_score @ 3
        self.all_score_weights = [0.60, 0.30, 0.05, 0.05]  #[0.7, 0.15, 0.15][0.60, 0.30, 0.05, 0.05]  #TODO negative weights??
        self.pos_switch = {
            'NN': 0.30,
            'NNS': 0.25,
            'NNP': 0.80,
            'NNPS': 0.70,
            'NG': 0.70
        }
        self.z_area_switch = {
            0: 0.90,
            1: 0.60,
            2: 0.30,
            3: 0.50  # n-gram
        }
        self.keyword_n = 7
        self.min_keyword_freq = 1