from keras.callbacks import Callback
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class Metrics(Callback):

    def on_epoch_end(self, epoch, logs={}):
        """Compute f1 score"""

        # this is what we passed to fit()
        valid_x = self.validation_data[0]
        valid_y = self.validation_data[1]

        # probability for each class (test size, num of classes)
        distribution = self.model.predict(valid_x)

        # turn into an indicator matrix
        distribution[distribution < 0.5] = 0
        distribution[distribution >= 0.5] = 1

        p = precision_score(valid_y, distribution, average='macro')
        r = recall_score(valid_y, distribution, average='macro')
        f1 = f1_score(valid_y, distribution, average='macro')
        print("precision: %.3f - recall: %.3f - f1: %.3f" % (p, r, f1))