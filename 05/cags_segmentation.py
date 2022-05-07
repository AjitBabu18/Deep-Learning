#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
from cags_segmentation_eval import CAGSMaskIoU
from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=30, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--finetuned_model", default="model_segment_3_finetuned.h5")
parser.add_argument("--model", default="model_segment_3.h5", type=str, help="Output model path.")
parser.add_argument("--tuning_epochs", default=1, type=int)
parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")


def train_augment(image, mask):
    # Horizontal flip with probability 0.5
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Zooming of the image needs to be done this way
    '''
    image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
    image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
                                    tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
    image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
    '''

    # Adjust the hue (odstín) of RGB images by a random factor.
    image = tf.image.random_hue(image, 0.08)
    # Adjust the saturation of RGB images by a random factor.
    image = tf.image.random_saturation(image, 0.6, 1.6)
    # Adjust the brightness of images by a random factor.
    image = tf.image.random_brightness(image, 0.05)
    # Adjust the contrast of an image or images by a random factor.
    image = tf.image.random_contrast(image, 0.7, 1.3)

    return image, mask

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    train_dataset = cags.train.map(lambda example: (example["image"], example["mask"]))
    dev_dataset = cags.dev.map(lambda example: (example["image"], example["mask"]))
    test_dataset = cags.test.map(lambda example: (example["image"], example["mask"]))
    
    train_data_pipeline = train_dataset.shuffle(1000, seed=args.seed)
    train_data_pipeline = train_data_pipeline.map(train_augment)
    train_data_pipeline = train_data_pipeline.batch(args.batch_size)
    
    dev_data_pipeline = dev_dataset.batch(args.batch_size)
    test_data_pipeline = test_dataset.batch(args.batch_size)
    
    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    
    efficientnet_b0.trainable = False
    
    inputs = tf.keras.layers.Input([CAGS.H, CAGS.W, CAGS.C])
    
    features = efficientnet_b0(inputs)
    
    x = features[1]
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    for feature in features[2:]:
        x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        f = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(feature)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.ReLU()(f)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        x = tf.keras.layers.Add()([x, f])
        
    outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same',
                                              activation=tf.nn.sigmoid, kernel_regularizer = 'l2')(x)

    # TODO: Create the model and train it
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(4e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalCrossentropy(name="accuracy")],
    )
    #tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.summary()

    # Train the model
    model.fit(x=train_data_pipeline, epochs=args.epochs,
              validation_data=dev_data_pipeline,
              #callbacks=[tb_callback]
              )

    # Save the model
    model.save(args.model)
    
    efficientnet_b0.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalCrossentropy(name="accuracy")],
    )
    #tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.summary()
    model.fit(x=train_data_pipeline, epochs=args.tuning_epochs, initial_epoch=args.epochs,
              validation_data=dev_data_pipeline,
              #callbacks=[tb_callback]
              )
    model.save(args.finetuned_model)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        test_masks = model.predict(test_data_pipeline)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)