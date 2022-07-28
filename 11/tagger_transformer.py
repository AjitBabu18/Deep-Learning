#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=800, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.layers = [
                tf.keras.layers.Dense(dim * expansion, activation=tf.nn.relu),
                tf.keras.layers.Dense(dim)
            ]
            #raise NotImplementedError()

        def get_config(self):
            return {"dim": self.dim, "expansion": self.expansion}

        def call(self, inputs):
            # TODO: Execute the FFN Transformer layer.
            x = inputs
            for layer in self.layers:
                x = layer(x)
            #print('x',x)
            return x
            #raise NotImplementedError()

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; for other arguments, keep the default values
            # (which mean trainable float32 matrices initialized with `"glorot_uniform"`).
            self.W_Q = self.add_weight(name="W_Q", shape=[dim, dim])
            self.W_K = self.add_weight(name="W_K", shape=[dim, dim])
            self.W_V = self.add_weight(name="W_V", shape=[dim, dim])
            self.W_O = self.add_weight(name="W_O", shape=[dim, dim])
            #raise NotImplementedError()

        def get_config(self):
            return {"dim": self.dim, "heads": self.heads}

        def call(self, inputs, mask):
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to [batch_size, max_sentence_len, heads, dim // heads],
            # - transpose via `tf.transpose` to [batch_size, heads, max_sentence_len, dim // heads].

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.

            # TODO: Apply the softmax, but including a suitable mask, which ignores all padding words.
            # The original `mask` is a bool matrix of shape [batch_size, max_sentence_len]
            # indicating which words are valid (True) or padding (False).
            # - You can perform the masking manually, by setting the attention weights
            #   of padding words to -1e9.
            # - Alternatively, you can use the fact that tf.keras.layers.Softmax accepts a named
            #   boolean argument `mask` indicating the valid (True) or padding (False) elements.

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - transpose the result to [batch_size, max_sentence_len, heads, dim // heads],
            # - reshape to [batch_size, max_sentence_len, dim],
            # - multiply the result by the W_O matrix.
            Q = inputs @ self.W_Q
            K = inputs @ self.W_K
            V = inputs @ self.W_V

            batch_size, max_sentence_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
            Q = tf.reshape(Q, shape=[batch_size, max_sentence_len, self.heads, self.dim // self.heads])
            K = tf.reshape(K, shape=[batch_size, max_sentence_len, self.heads, self.dim // self.heads])
            V = tf.reshape(V, shape=[batch_size, max_sentence_len, self.heads, self.dim // self.heads])

            Q = tf.transpose(Q, perm=[0, 2, 1, 3])
            K = tf.transpose(K, perm=[0, 2, 1, 3])
            V = tf.transpose(V, perm=[0, 2, 1, 3])
            
            SA = Q @ tf.transpose(K, perm=[0, 1, 3, 2])
            norm = tf.math.sqrt(tf.cast(self.dim // self.heads, tf.float32))
            SA = SA / norm
            
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.expand_dims(mask, axis=2)
            #print(mask)
            self_attention = tf.keras.layers.Softmax(axis=3)(SA, mask=mask)
            
            result = self_attention @ V
            result = tf.transpose(result, perm=[0, 2, 1, 3])
            result = tf.reshape(result, shape=[batch_size, max_sentence_len, self.dim])
            result = result @ self.W_O

            return result
            
            #raise NotImplementedError()

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # TODO: Create the required number of transformer layers, each consisting of
            # - a layer normalization and a self-attention layer followed by a dropout layer,
            # - a layer normalization and a FFN layer followed by a dropout layer.
            
            self.layer_SA = []
            self.layer_SA_dropout = []
            self.layer_SA_norm = []
            self.layer_ffn = []
            self.layer_ffn_dropout = []
            self.layer_ffn_norm = []

            for arr, constructor in [(self.layer_SA_norm, lambda: tf.keras.layers.LayerNormalization()),
                           (self.layer_SA, lambda: Model.SelfAttention(dim, heads)),
                           (self.layer_SA_dropout, lambda: tf.keras.layers.Dropout(dropout)),
                           
                           (self.layer_ffn_norm, lambda: tf.keras.layers.LayerNormalization()),
                           (self.layer_ffn, lambda: Model.FFN(dim, expansion)),
                           
                           (self.layer_ffn_dropout, lambda: tf.keras.layers.Dropout(dropout))
                            ]:
                for i in range(self.layers):
                    #print(constructor)
                    arr.append(constructor())

        def get_config(self):
            return {name: getattr(self, name) for name in ["layers", "dim", "expansion", "heads", "dropout"]}

        def call(self, inputs, mask):
            # TODO: Start by computing the sinusoidal positional embeddings.
            # They have a shape `[max_sentence_len, dim]` and
            # - for `i < dim / 2`, the value on index `[pos, i]` should be
            #     `sin(pos / 10000 ** (2 * i / dim))`
            # - the value on index `[pos, i]` for `i >= dim / 2` should be
            #     `cos(pos / 10000 ** (2 * (i - dim/2) / dim))`
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layer, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            max_sentence_len = tf.shape(inputs)[1]
            indexes_0th_dim = tf.range(max_sentence_len, dtype=tf.float32)
            indexes_1th_dim = tf.range(self.dim/2, dtype=tf.float32)

            top = tf.map_fn(lambda i: tf.math.sin(indexes_0th_dim / 10000 ** (2 * i / self.dim)), indexes_1th_dim)
            bottom = tf.map_fn(lambda i: tf.math.cos(indexes_0th_dim / 10000 ** (2 * i / self.dim)), indexes_1th_dim)

            top = tf.transpose(top)
            bottom = tf.transpose(bottom)

            embeddings = tf.concat([top, bottom], axis=1)
            
            x = inputs + embeddings

            for sa, sa_dropout, sa_norm, ffn, ffn_dropout, ffn_norm in zip(self.layer_SA, self.layer_SA_dropout, self.layer_SA_norm,
                                                                           self.layer_ffn, self.layer_ffn_dropout, self.layer_ffn_norm):
                residual = x
                x = sa_norm(residual)
                x = sa(x, mask=mask)
                x = sa_dropout(x)
                #residual = x
                #x = sa_norm(x + residual)
                x = x + residual
                residual = x
                x = ffn_norm(residual)
                x = ffn(x)
                x = ffn_dropout(x)
                #x = ffn_norm(x + residual)
                #print(x)
                #
                #x = ffn_norm(x + residual)
                x = x + residual

            return x
            #raise NotImplementedError()

    def __init__(self, args, train):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.

        mapped_words = train.forms.word_mapping(words)
        

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        embedded_words = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(mapped_words)

        # TODO: Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        transformer_layer = Model.Transformer(args.transformer_layers, args.we_dim, args.transformer_expansion,
                                                args.transformer_heads, args.transformer_dropout)
        
        transformer_layer = transformer_layer(embedded_words.to_tensor(), mask=tf.sequence_mask(embedded_words.row_lengths()))
        transformer_layer = tf.RaggedTensor.from_tensor(transformer_layer, embedded_words.row_lengths())

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        tags = train.tags.word_mapping.vocabulary_size()
        predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tags, activation=tf.nn.softmax))(transformer_layer)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=lambda yt, yp: tf.losses.SparseCategoricalCrossentropy()(yt.values, yp.values),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> Dict[str, float]:
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

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # TODO(tagger_we): Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integral tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.
    def extract_tagging_data(example):
        forms = example["forms"]
        tags = example["tags"]
        return forms, morpho.train.tags.word_mapping(tags)
        #raise NotImplementedError()

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development and training losses for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
