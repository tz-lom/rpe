from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pickle
import sys
import json
import operator
import os


def read_params(configFileName):
    #config = path_to_config
    #if len(sys.argv) == 2:
    #    config = sys.argv[1]
    #path = os.getcwd()
    #parentPath = os.path.join(path, os.pardir)
    #path_to_config = os.path.join(parentPath, configFileName)

    path_to_config = os.path.join(os.path.dirname(__file__), configFileName)

    with open(path_to_config) as f:
        config = json.load(f)
    return config

def write_params(configFileName, key, param):
    path_to_config = os.path.join(os.path.dirname(__file__), configFileName)
    with open(path_to_config) as f:
        data = json.load(f)
    data[key] = param
    with open(path_to_config, 'w') as fp:
        json.dump(data, fp)


class AucMetricHistory(Callback):
    def __init__(self,validation_data,save_best_by_auc=False,path_to_save=None):
        super(AucMetricHistory, self).__init__()
        self.validation_data = validation_data
        self.save_best_by_auc=save_best_by_auc
        self.path_to_save = path_to_save
        self.best_auc = 0
        self.best_epoch = 1
        if save_best_by_auc and (path_to_save is None):
            raise ValueError('Specify path to save the model')

    def on_epoch_end(self, epoch, logs={}):
        x_val,y_val = self.validation_data[0],self.validation_data[1]
        y_pred = self.model.predict(x_val,batch_size=len(y_val), verbose=0)
        if isinstance(y_pred,list):
            y_pred = y_pred[0]
        current_auc = roc_auc_score(y_val, y_pred)
        logs['val_auc'] = current_auc

        if current_auc > self.best_auc:
            self.best_auc = current_auc
            self.best_epoch = epoch


def single_auc_loging(history, title, path_to_save):
    """
    Function for ploting nn-classifier performance. It makes two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plot as a picture and as a pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param path_to_save: Path to save file
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))

    if 'loss' in history.keys():
        loss_key = 'loss'  # for simple NN
    elif 'class_out_loss' in history.keys():
        loss_key = 'class_out_loss'  # for DAL NN
    else:
        raise ValueError('Not found correct key for loss information in history')

    ax1.plot(history[loss_key], label='cl train loss')
    ax1.plot(history['val_%s' % loss_key], label='cl val loss')
    ax1.legend()
    min_loss_index, max_loss_value = min(enumerate(history['val_loss']), key=operator.itemgetter(1))
    ax1.set_title('min_loss_%.3f_epoch%d' % (max_loss_value, min_loss_index))
    ax2.plot(history['val_auc'])
    max_auc_index, max_auc_value = max(enumerate(history['val_auc']), key=operator.itemgetter(1))
    ax2.set_title('max_auc_%.3f_epoch%d' % (max_auc_value, max_auc_index))
    f.suptitle('%s' % (title))
    plt.savefig('%s/%s.png' % (path_to_save, title))
    plt.close()
    with open('%s/%s.pkl' % (path_to_save, title), 'wb') as output:
        pickle.dump(history, output, pickle.HIGHEST_PROTOCOL)

