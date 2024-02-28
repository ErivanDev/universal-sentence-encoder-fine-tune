import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
# import seaborn as sns

X = [
  # Smartphones
  "I like my phone",
  "My phone is not good.",
  "Your cellphone looks great.",

  # Weather
  "Will it snow tomorrow?",
  "Recently a lot of hurricanes have hit the US",
  "Global warming is real",

  # Food and health
  "An apple a day, keeps the doctors away",
  "Eating strawberries is healthy",
  "Is paleo better than keto?",

  # Asking about age
  "How old are you?",
  "what is your age?",
]

y = [0,0,0,1,1,1,2,2,2,3,3]

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  '''sns.set(font_scale=1.2)
  g = sns.heatmap(
    corr,
    xticklabels=labels,
    yticklabels=labels,
    vmin=0,
    vmax=1,
    cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")'''

  np.savetxt('main.txt', corr)

def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
  message_embeddings_ = session_.run(
    encoding_tensor, feed_dict={input_tensor_: messages_})
  plot_similarity(messages_, message_embeddings_, 90)

from tensorflow.python.saved_model import tag_constants

scope = 'finetune'

graph=tf.Graph()

with tf.Session(graph=graph) as sess:
  model_path = 'model/'

  tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

  sess.run(tf.global_variables_initializer())
  sess.run(tf.get_default_graph().get_operation_by_name('finetune/init_all_tables'))

  in_tensor = tf.get_default_graph().get_tensor_by_name(scope + '/module/fed_input_values:0')
  ou_tensor = tf.get_default_graph().get_tensor_by_name(scope + '/module/Encoder_en/hidden_layers/l2_normalize:0')

  run_and_plot(sess, in_tensor, X, ou_tensor)