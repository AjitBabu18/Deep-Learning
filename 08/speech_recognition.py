#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--ctc_beam", default=10, type=int, help="size")
parser.add_argument("--rnn_cell_dim", default=256, type=int, help="RNN cell dimension.")

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        inputs = tf.keras.layers.Input(shape=[None, CommonVoiceCs.MFCC_DIM], dtype=tf.float32, ragged=True)

        # TODO: Create a suitable model. You should:
        # - use a bidirectional RNN layer(s) to contextualize the input sequences.
        #
        # - optionally use suitable regularization
        #
        # - and finally generate logits for CTC loss/prediction as RaggedTensors.
        #   The logits should be generated by a dense layer with `1 + len(CommonVoiceCs.LETTERS)`
        #   outputs (the plus one is for the CTC blank symbol). Note that no
        #   activation should be used (the CTC operations will take care of it).
        
        bidirectional_0 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True),merge_mode='sum')(inputs.to_tensor(),mask=tf.sequence_mask(inputs.row_lengths()))
        bidirectional = tf.RaggedTensor.from_tensor(bidirectional_0, inputs.row_lengths())
        logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(CommonVoiceCs.LETTERS) + 1, activation=None))(bidirectional)

        super().__init__(inputs=inputs, outputs=logits)
        
        lr_decay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001, end_learning_rate=0.0001, decay_steps=9773 / args.batch_size * args.epochs)

        # We compile the model with the CTC loss and EditDistance metric.
        self.compile(optimizer=tf.optimizers.Adam(lr_decay, tf.constant(0, tf.int32)),
                     loss=self.ctc_loss,
                     metrics=[CommonVoiceCs.EditDistanceMetric()])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # TODO: Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the `gold_labels` to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tf.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.
        gold_labels.to_tensor()
        gold_labels = gold_labels.to_sparse()
        l = logits.row_lengths()
        logits = logits.to_tensor()
        logits = tf.transpose(logits, [1, 0, 2])
        loss = tf.nn.ctc_loss(tf.cast(gold_labels, dtype=tf.int32), logits, label_length=None, logit_length=tf.cast(l, dtype=tf.int32), blank_index=len(CommonVoiceCs.LETTERS))
        return tf.reduce_mean(loss)
        
        #raise NotImplementedError()

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # TODO: Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor
        l = logits.row_lengths()
        sequence_length = tf.cast(l, tf.int32)
        
        logits = logits.to_tensor()
        logits = tf.transpose(logits, [1, 0, 2])
        
        (predictions,), _ = tf.nn.ctc_beam_search_decoder(logits, sequence_length, beam_width=args.ctc_beam)
        
        predictions = tf.RaggedTensor.from_sparse(predictions)

        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        #y_pred = y_pred.astype(np.int32)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)


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

    # Load the data.
    cvcs = CommonVoiceCs()

    # Create input data pipeline.
    def create_dataset(name):
        def prepare_example(example):
            # TODO: Create suitable batch examples.
            # - example["mfccs"] should be used as input
            # - the example["sentence"] is a UTF-8-encoded string with the target sentence
            #   - split it to unicode characters by using `tf.strings.unicode_split`
            #   - then pass it through the `cvcs.letters_mapping` layer to map
            #     the unicode characters to ids
            inputs, outputs = example["mfccs"], example["sentence"]
            outputs = tf.strings.unicode_split(outputs, 'UTF-8')
            outputs = cvcs.letters_mapping(outputs)
            outputs = tf.cast(outputs, dtype=tf.int32)
            return inputs, outputs
            
            #raise NotImplementedError()

        dataset = getattr(cvcs, name).map(prepare_example)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    # TODO: Create the model and train it
    model = Model(args)
    model.fit(train, epochs=args.epochs, validation_data=dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS[char] for char in sentence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    #print(type(args.ctc_beam))
    main(args)
