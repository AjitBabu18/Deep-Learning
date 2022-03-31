#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=80, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=5, type=int, help="Window size to use.")
parser.add_argument("--dropout", default=0.4, type=float, help="dropout")
parser.add_argument("--hidden_layers", default=1024, type=int, help="Hidden layer count")
parser.add_argument("--l2", default=None, type=float, help="l2 regularization")
parser.add_argument("--train", default=True, help="Train or not")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    args.save_dir = "logs"
    args.base_save_name = "weights.best-2.hdf5"
    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is represented by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    #model = crateModel(args, uppercase_data)
    """model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32))
    model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
    model.add(tf.keras.layers.Flatten())
    if args.dropout is not None:
        model.add(tf.keras.layers.Dropout(args.dropout))
    for hidden_layer in args.hidden_layers:
        if args.dropout is not None:
            model.add(tf.keras.layers.Dropout(args.dropout))
        l2_regularizer = None
        if args.l2 is not None:
            l2_regularizer = tf.keras.regularizers.l1_l2(l2=args.l2)

        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu, kernel_regularizer=l2_regularizer))
    l2_regularizer = None
    if args.l2 is not None:
        l2_regularizer = tf.keras.regularizers.L1L2(l2=args.l2)
    model.add(tf.keras.layers.Dense(UppercaseData.LABELS, activation=tf.nn.softmax, kernel_regularizer=l2_regularizer))"""
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[2*args.window +1], dtype=tf.int32))
    model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(args.dropout))
    l2_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=0)
    model.add(tf.keras.layers.Dense(args.hidden_layers, activation=tf.nn.relu, kernel_regularizer=l2_regularizer))
    model.add(tf.keras.layers.Dropout(args.dropout))
    
    if args.train:
        _loss= tf.keras.losses.SparseCategoricalCrossentropy()
        _accuracy= tf.metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=_loss,
            metrics=[_accuracy]
        )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.fit(uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
            callbacks=[tb_callback])

    scores = model.evaluate(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"])
    print("Dev accuracy is %.2f%%" % (scores[1] * 100))

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).
    os.makedirs(args.logdir, exist_ok=True)
    X = uppercase_data.test.data["windows"]
    #up = model.predict(x=X, batch_size=args.batch_size)

    # Load original text
    text = uppercase_data.test.text
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        for i in range(len(X)):
            char_from_text = text[i]
           
            if char_from_text is 'ß':
                #print(char_from_text, " ")
                print("%s" % (char_from_text), end="", file=predictions_file)

            else:
                print("%s" % (char_from_text.upper()), end="", file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
