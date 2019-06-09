import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, epsilon=0.0005):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            epsilon (float): If loss does not improve more than epsilon for patience*2 epochs, stop training
                            Default: 0.0005
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.counterEps = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.eps = epsilon

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            self.counterEps += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            if self.counterEps >= self.patience * 2:
                print(f'Early stopping due to no significant improvement of validation loss')
                self.early_stop = True
        else:
            if score-self.best_score < self.eps:
                self.counterEps += 1
            else:
                self.counterEps = 0
            if self.counterEps >= self.patience*2:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                print(f'Early stopping due to no significant improvement of validation loss')
                self.early_stop = True
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss