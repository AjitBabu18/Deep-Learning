#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--tuning_epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--dropout", default=0.4, type=int, help="Drop out rate.")
parser.add_argument("--model", default="model_3.h5", type=str, help="Output model path.")
parser.add_argument("--finetuned_model", default="model_3_finetuned.h5", type=str, help="Output model path.")
parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")



def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    train_dataset = cags.train.map(lambda example: (example["image"], example["label"]))
    dev_dataset = cags.dev.map(lambda example: (example["image"], example["label"]))
    test_dataset = cags.test.map(lambda example: (example["image"], example["label"]))
    
    train_data_pipeline = train_dataset.shuffle(1000, seed=args.seed)
    #train_data_pipeline = train_data_pipeline.map(train_augment)
    train_data_pipeline = train_data_pipeline.batch(args.batch_size)
    
    dev_data_pipeline = dev_dataset.batch(args.batch_size)
    test_data_pipeline = test_dataset.batch(args.batch_size)
    
    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    efficientnet_b0.trainable = False
    
    inputs = tf.keras.layers.Input([CAGS.H, CAGS.W, CAGS.C])
    hidden = efficientnet_b0(inputs)[0]
    hidden = tf.keras.layers.Dropout(rate=0.2)(hidden)
    hidden = tf.keras.layers.Dense(500, activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(rate=0.3)(hidden)
    hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(rate=0.4)(hidden)
    hidden = tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax)(hidden)
    
    # TODO: Create the model and train it
    model = tf.keras.Model(inputs=inputs, outputs=hidden)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(4e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.summary()

    model.fit(x=train_data_pipeline, epochs=args.epochs,
              validation_data=dev_data_pipeline,
              callbacks=[tb_callback]
              )

    model.save(args.model)
    
    efficientnet_b0.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    model.summary()

    model.fit(x=train_data_pipeline, epochs=args.tuning_epochs, initial_epoch=args.epochs,
              validation_data=dev_data_pipeline,
              callbacks=[tb_callback]
              )

    model.save(args.finetuned_model)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test_data_pipeline)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)