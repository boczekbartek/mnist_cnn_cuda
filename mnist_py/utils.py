import os
import logging
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import History

__all__ = ['save_plots', 'save_model']


def create_dump_dir(name: str):
    here = os.path.abspath(os.path.dirname(__file__))
    models_path = os.path.join(here, name)
    os.makedirs(models_path, exist_ok=True)
    return models_path


def save_plots(history: History, model_name: str):
    figures_dir = create_dump_dir('figures_dump')
    acc_figure_fname = os.path.join(figures_dir, f'{model_name}.acc.png')
    loss_figure_fname = os.path.join(figures_dir, f'{model_name}.loss.png')


    def plot_metrics(keys, save_fname):
        plt.figure()
        for k in keys:
            plt.plot(history.history[k])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(save_fname)

    plot_metrics(('acc', 'val_acc'), acc_figure_fname)
    plot_metrics(('loss', 'val_loss'), loss_figure_fname)


def save_model(model: Sequential, model_name: str):
    model_name = f'model.{model_name}'
    models_path = create_dump_dir('models_dump')
    logging.info(f"Saving model to {models_path}")
    model.save(os.path.join(models_path, model_name))
    logging.info("Model saved")
