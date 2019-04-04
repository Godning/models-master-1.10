# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines NeuMF model for NCF framework.

Some abbreviations used in the code base:
NeuMF: Neural Matrix Factorization
NCF: Neural Collaborative Filtering
GMF: Generalized Matrix Factorization
MLP: Multi-Layer Perceptron

GMF applies a linear kernel to model the latent feature interactions, and MLP
uses a nonlinear kernel to learn the interaction function from data. NeuMF model
is a fused model of GMF and MLP to better model the complex user-item
interactions, and unifies the strengths of linearity of MF and non-linearity of
MLP for modeling the user-item latent structures.

In NeuMF model, it allows GMF and MLP to learn separate embeddings, and combine
the two models by concatenating their last hidden layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from official.datasets import movielens  # pylint: disable=g-bad-import-order
from official.recommendation.transform import multihead_attention, feedforward
"""Neural matrix factorization (NeuMF) model for recommendations."""


def model_fn(features, labels, mode, params):
    """Initialize NeuMF model.

    Args:
      num_users: An integer, the number of users.
      num_items: An integer, the number of items.
      mf_dim: An integer, the embedding size of Matrix Factorization (MF) model.
      model_layers: A list of integers for Multi-Layer Perceptron (MLP) layers.
        Note that the first layer is the concatenation of user and item
        embeddings. So model_layers[0]//2 is the embedding size for MLP.
      batch_size: An integer for the batch size.
      mf_regularization: A floating number, the regularization factor for MF
        embeddings.
      mlp_reg_layers: A list of floating numbers, the regularization factors for
        each layer in MLP.

    Raises:
      ValueError: if the first model layer is not even.
    """

    num_users = params["num_users"]
    num_items = params["num_items"]
    fm_dim = params["mf_dim"]
    model_layers = params["model_layers"]
    mf_regularization = params["mf_regularization"]
    mlp_reg_layers = params["mlp_reg_layers"]
    lr = params["lr"]
    if model_layers[0] % 2 != 0:
        raise ValueError("The first layer size should be multiple of 2!")

    # Input variables

    user_input = features[movielens.USER_COLUMN]
    item_input = features[movielens.ITEM_COLUMN]
    # Initializer for embedding layer
    embedding_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    # Embedding layers of GMF and MLP
    fm_embedding_user = tf.keras.layers.Embedding(
        num_users,
        fm_dim,
        embeddings_initializer=embedding_initializer,
        embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
        input_length=1)(user_input)
    fm_embedding_item = tf.keras.layers.Embedding(
        num_items,
        fm_dim,
        embeddings_initializer=embedding_initializer,
        embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
        input_length=1)(item_input)



    # MF part
    # Flatten the embedding vector as latent features in MLP
    fm_user_latent = tf.keras.layers.Flatten()(fm_embedding_user)
    fm_item_latent = tf.keras.layers.Flatten()(fm_embedding_item)
    # Concatenation of two latent features
    concat_embedding = tf.keras.layers.concatenate([fm_user_latent, fm_item_latent])

    first_order = tf.reduce_sum(concat_embedding, axis=-1, keep_dims=True)

    square_of_sum = tf.square(tf.reduce_sum(concat_embedding, axis=1, keep_dims=True))
    sum_of_square = tf.reduce_sum(tf.square(concat_embedding), axis=1, keep_dims=True)
    second_order = 0.5 * tf.subtract(square_of_sum, sum_of_square)

    num_layer = len(model_layers)  # Number of layers in the MLP
    for layer in xrange(1, num_layer):
        model_layer = tf.keras.layers.Dense(
            model_layers[layer],
            kernel_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[layer]),
            activation="relu")
        mlp_vector = model_layer(concat_embedding)

    # Concatenate GMF and MLP parts
    predict_vector = tf.keras.layers.concatenate([first_order, second_order, mlp_vector])

    # Final prediction layer
    prediction = tf.keras.layers.Dense(
        1, kernel_initializer="lecun_uniform",
        name=movielens.RATING_COLUMN)(predict_vector)

    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.cast(labels, tf.float32)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=labels)
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            movielens.RATING_COLUMN: tf.nn.sigmoid(prediction)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
