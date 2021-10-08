from typing import *

import tensorflow as tf
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.keras.callbacks import Callback


class EpochCheckpointManagerCallback(Callback):
    """
    This allows using the TF Checkpoint & CheckpointManager API in model.fit()
    instead of having to override the training loop.

    You can use .save_counter on the checkpoint (after restoring) to get the 
    last epoch.

    Example usage:
    ```python
    model = build_model()

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, "saved/", max_to_keep=2)
    checkpoint_callback = cb_extra.EpochCheckpointManagerCallback(manager)

    latest = manager.latest_checkpoint
    initial_epoch = 0
    if latest:
        print("Using model checkpoint {}".format(latest))

        # We add .expect_partial(), because if a previous run completed,
        # and we consequently restore from the last checkpoint, no further
        # training is need, and we don't expect to use all variables.
        checkpoint.restore(latest).expect_partial()
        initial_epoch = int(checkpoint.save_counter)
    else:
        print("Training from scratch")

    model.fit(
        train,
        epochs=6,
        validation_data=test,
        callbacks=[checkpoint_callback],
        initial_epoch=initial_epoch
    )
    ```

    Sources:
    [1] https://www.tensorflow.org/guide/checkpoint
    [2] https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    [3] https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager
    [4] https://www.tensorflow.org/guide/keras/custom_callback
    """

    def __init__(self,
                 manager: CheckpointManager,
                 epoch: Optional[tf.Variable] = None,
                 verbose: bool = False):
        self.manager = manager
        self.epoch = epoch
        self.verbose = verbose
        super().__init__()

    def on_epoch_end(self, epoch, logs):
        if self.epoch is not None:
            self.epoch.assign_add(1)

        save_path = self.manager.save()
        if self.verbose:
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
        return super().on_epoch_end(epoch, logs=logs)
