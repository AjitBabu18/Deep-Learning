#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import itertools
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
try:
    import transformers
except Exception:
    raise RuntimeError("You need to install the `transformers` package")
#from eleczech_lc_small import ElectraCzechSmallLc

from text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=16, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.00005, type=float, help="lr")
parser.add_argument("--flr", default=0.00005, type=float, help="Fine tuning learning rate")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization")

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the Electra Czech small lowercased
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small", output_hidden_states=True)
    

    # TODO: Load the data. Consider providing a `tokenizer` to the
    # constructor of the TextClassificationDataset.
    facebook = TextClassificationDataset("czech_facebook", tokenizer=tokenizer.encode)
    
    def prepare(dataset: TextClassificationDataset.Dataset, test=False):

        if test:
            ds = dataset.dataset.map(lambda x: x['tokens'])
        else:
            ds = dataset.dataset.map(lambda x: (x['tokens'], facebook.train.label_mapping(x['labels'])))

        if not test:
            ds = ds.shuffle(512)

        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return ds

    def calculate_max_sentence_len(train, dev, test):
        max_sentence_length = 0

        for i in itertools.chain(
                train.map(lambda x, y: tf.math.reduce_max(x.row_lengths())),
                dev.map(lambda x, y: tf.math.reduce_max(x.row_lengths())),
                test.map(lambda x: tf.math.reduce_max(x.row_lengths()))):

            if i > max_sentence_length:
                max_sentence_length = i

        return tf.cast(max_sentence_length, tf.int32)

    train = prepare(facebook.train)
    dev = prepare(facebook.dev)
    test = prepare(facebook.test, test=True)
    max_sentence_length = calculate_max_sentence_len(train, dev, test)

    #print(max_sentence_length)

    # TODO: Create the model and train it
    inputs = tf.keras.layers.Input(shape=[None], ragged=True)

    padding_size = max_sentence_length - tf.shape(inputs.to_tensor())[1]
    padding = tf.zeros([tf.shape(inputs.to_tensor())[0], padding_size], tf.int32)

    inputs_tensor = tf.cast(inputs.to_tensor(), tf.int32)
    dense_inputs = tf.concat([inputs_tensor, padding], axis=1)
    

    sequence_lengths = inputs.row_lengths()
    mask = tf.sequence_mask(sequence_lengths, maxlen=max_sentence_length, dtype=tf.int32)

    x = eleczech(dense_inputs, attention_mask=mask)
    #print(x.hidden_states[-12:])
    x = tf.concat(x.hidden_states[-12:], axis=-1)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(args.l2))(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)

    # TODO: Create the model and train it
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    eleczech.trainable = False
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy()
    )
    
    
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    eleczech.trainable = True
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=args.flr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy()
    )

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test)

        label_strings = facebook.test.label_mapping.get_vocabulary()
        for sentence in predictions:
            print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
