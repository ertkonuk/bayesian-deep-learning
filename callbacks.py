# . . these callbacks are inspired by the torchsample project
# . . see the link for more:
# . . https://github.com/ncullen93/torchsample
import torch 
import numpy as np
import datetime
import os
# . . utility functions
def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")

# . . keras-like callbacks for pytorch training
# . . holds and executes a list of callbacks
class CallbackHandler(object):

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def extend(self, callbacks):
        self.callbacks.extend(callbacks)

    def on_epoch_begin(self, epoch, logs=None, model=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs, model)

    def on_epoch_end(self, epoch, logs=None, model=None):
        logs = logs or {}        
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs, model)

    def on_batch_begin(self, batch, logs=None, model=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs, model)

    def on_batch_end(self, batch, logs=None, model=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs, model)

    def on_train_begin(self, logs=None, model=None):
        logs = logs or {}
        logs['start_time'] = _get_current_time()
        print('Training started: ', logs['start_time'])
        for callback in self.callbacks:
            callback.on_train_begin(logs, model)

    def on_train_end(self, logs=None, model=None):
        logs = logs or {}
        #logs['final_loss'] = self.trainer.history.epoch_losses[-1],
        #logs['best_loss'] = min(self.trainer.history.epoch_losses),
        logs['stop_time'] = _get_current_time()
        print('Training finished: ', logs['stop_time'])    
        for callback in self.callbacks:
            callback.on_train_end(logs, model)


# . . the abstract callback base class
class Callback(object):

    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None, model=None):
        pass

    def on_epoch_end(self, epoch, logs=None, model=None):
        pass

    def on_batch_begin(self, batch, logs=None, model=None):
        pass

    def on_batch_end(self, batch, logs=None, model=None):
        pass

    def on_train_begin(self, logs=None, model=None):
        pass

    def on_train_end(self, logs=None, model=None):
        pass


# . . callback that prints the loss after every epoch
class EpochMetrics(Callback):
    def __init__(self, monitor='loss', skip=1):

        self.monitor = monitor
        # . . print at every skip-th epoch
        self.skip = skip

        self.start_time = 0.
        self.end_time = 0.
        self.elapsed_time = 0.
        # . . call the parent(abstract)
        super(EpochMetrics, self).__init__()

    def on_epoch_begin(self, epoch, logs=None, model=None): 
        self.start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None, model=None):        
        self.end_time = datetime.datetime.now()
        # . . compute the time for a single epoch
        self.elapsed_time = self.end_time - self.start_time
        if (epoch % self.skip) == 0:           
            print('Epoch: {:5d}\tTrain Loss: {:.4f}\tValid Loss: {:.4f}\tKLDiv Loss: {:.4f}\tTrain Accuracy: {:.4f}\tValid Accuracy: {:.4f}\tElapsed time: {}'. format(epoch, logs.get('train_loss'), logs.get('valid_loss'), logs.get('kldiv_loss'), logs.get('train_acc'), logs.get('valid_acc'), self.elapsed_time))

# . . callback that prints the loss after every batch
class BatchMetrics(Callback):
    def __init__(self, monitor='batch_loss', skip=1):

        self.monitor = monitor
        # . . print at every skip-th epoch
        self.skip = skip

        # . . call the parent(abstract)
        super(BatchMetrics, self).__init__()

    def on_batch_end(self, batch, logs=None, model=None):        
        if (batch % self.skip) == 0:           
            print('Batch: {:5d} of {:5d}\tBatch Loss: {:.6f}'. format(batch, logs.get('num_train_batch'), logs.get(self.monitor)))

# . . saves the best model during the training and returns it
class ReturnBestModel(Callback):
    def __init__(self, monitor='valid_loss', min_delta=0, skip=1, reset=True, path='models/',):
        # . . first run?
        self.reset = reset
        # . . watch the monitor
        self.monitor = monitor 
        # . . minimum loss reduction to be considered as improvement
        self.min_delta = min_delta
        # . . check every skip-th epoch
        self.skip = skip
        # . . keep track of the best loss
        self.best_loss = 0.
        self.best_loss_epoch = 0
        # . . set the path
        self.path = path
        # . . set the best model path
        self.bestmodelpath = self.path + '/best_model.pt'

        # . . call the parent(abstract)
        super(ReturnBestModel, self).__init__()

    # . . save the trainable model parameters
    def save_model(self, model):
        torch.save(model.state_dict(), self.bestmodelpath)

    # . . load the best model (weights)
    def load_model(self, model):
        model.load_state_dict(torch.load(self.bestmodelpath))

    def on_train_begin(self, logs=None, model=None):
        if self.reset: 
            self.best_loss = 1e15
            # . . delete the best model file: ugly use of ternaries!
            if os.path.exists(self.bestmodelpath): os.remove(self.bestmodelpath)
            if not os.path.exists(self.path): os.makedirs(self.path)

    def on_epoch_end(self, epoch, logs=None, model=None):
        current_loss = logs.get(self.monitor)

        if (epoch % self.skip) == 0:
            # . . if the current loss is lower
            if (self.best_loss - current_loss) > self.min_delta:
                #  . . get the new best loss
                self.best_loss = current_loss
                # . . get the epoch of the best loss
                self.best_loss_epoch = epoch
                # . . register to log
                logs["best_loss"] = self.best_loss
                # . . save the model weights
                self.save_model(model)

    def on_train_end(self, logs=None, model=None):
        # . . print the best loss
        print('Best Loss: {:.6f}\tEpoch: {:5d}'. format(self.best_loss, self.best_loss_epoch))
        # . . load the best model
        self.load_model(model)


# . . exit the training loop if the training or validation loss does not improve
class EarlyStopping(Callback):

    def __init__(self, monitor='valid_loss', min_delta=0, patience=5):

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0

        super(EarlyStopping, self).__init__()

    def on_train_begin(self, logs=None, model=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, logs=None, model=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            if (self.best_loss - current_loss) >= self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    model._stop_training = True
                self.wait += 1
        
    def on_train_end(self, logs=None, model=None):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %05i' % 
                (self.stopped_epoch))

# . . incomplete
# . . Callback that terminates training when a NaN loss is encountered.
#class TerminateOnNaN(Callback):
#
#  def on_batch_end(self, batch, logs=None, model=None):
#    logs = logs or {}
#    loss = logs.get('loss')
#    if loss is not None:
#      if np.isnan(loss) or np.isinf(loss):
#        print('Batch %d: Invalid loss, terminating training' % (batch))
#        model.stop_training = True