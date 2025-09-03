import sys
import os
import tensorflow as tf


def create_batcher(input_ids, batch_size= 32, shuffle=True, buffer_size= 10000):
    input_ids = tf.convert_to_tensor(input_ids)

    ds = tf.data.Dataset.from_tensor_slices(input_ids)

    if shuffle:
        ds= ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds= ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

if __name__=="__main__":
    import numpy as np

    dummy= np.random.randint(0,1000, (100,6))
    batcher = create_batcher(dummy, batch_size=6)
    for batch in batcher.take(1):
        print(batch.shape)