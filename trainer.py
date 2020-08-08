import os
from tensorflow.keras.callbacks import ModelCheckpoint


class Trainer():
    def __init__(self, model, ckpt_dir, learning_rate, epochs):
        super(Trainer, self).__init__()
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        # self.result_dir = result_dir

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.ckpt_dir, 'ckpt-{epoch:04d}-{val_loss:.4f}.hdf5'),
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
            verbose=1
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])

    def test(self, test_batch_generator, test_steps_per_epoch):
        self.model.evaluate(test_batch_generator, steps=test_steps_per_epoch)
