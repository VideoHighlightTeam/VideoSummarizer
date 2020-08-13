import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from datetime import datetime as dt
from pytz import timezone


class Trainer:
    def __init__(self, model: Model, ckpt_dir, learning_rate, epochs, class_weight=None):
        # super(Trainer, self).__init__()
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight

        self.callbacks = []
        self.loss = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.val_loss = []
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.init_callbacks()

    def init_callbacks(self):
        kst = timezone('Asia/Seoul')
        ckpt_filename_format = 'ckpt-' + dt.now(tz=kst).strftime('%Y%m%d-%H%M%S') + '-{epoch:04d}-{val_loss:.4f}.hdf5'
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.ckpt_dir, ckpt_filename_format),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        )

    def train(self, train_batch_generator, train_steps_per_epoch, valid_batch_generator=None, valid_steps_per_epoch=None):
        history = self.model.fit(
            train_batch_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=valid_batch_generator,
            validation_steps=valid_steps_per_epoch,
            class_weight=self.class_weight,
            verbose=1
        )
        self.loss.extend(history.history['loss'])
        self.accuracy.extend(history.history['accuracy'])
        self.precision.extend(history.history['precision'])
        self.recall.extend(history.history['recall'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_accuracy.extend(history.history['val_accuracy'])
        self.val_precision.extend(history.history['val_precision'])
        self.val_recall.extend(history.history['val_recall'])

    def test(self, test_batch_generator, test_steps_per_epoch):
        self.model.evaluate(test_batch_generator, steps=test_steps_per_epoch)
