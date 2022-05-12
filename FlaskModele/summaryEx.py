from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
import nltk
import torch
import transformers as ppb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from nltk import sent_tokenize

from keras.preprocessing.sequence import pad_sequences

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import os


def main(paragraph, n_sentence, maxLength, n_topic):
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained("Ghani-25/SummFinFR")

    model = AutoModelForMaskedLM.from_pretrained("Ghani-25/SummFinFR")

    paragraph_split = tokenizer.tokenize(paragraph)  # split the paragraph

    input_tokens = []
    for i in paragraph_split:
        input_tokens.append(tokenizer.encode(i, add_special_tokens=True))

    temp = []
    for i in input_tokens:
        temp.append(len(i))
    np.max(temp)  # the longest sentence in our paragraph has 129 tokens.

    input_ids = pad_sequences(input_tokens, maxlen=100, dtype="long", value=0, truncating="post", padding="post")

    def create_attention_mask(input_id):
        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]  # create a list of 0 and 1.
            attention_masks.append(att_mask)  # basically attention_masks is a list of list
        return attention_masks

    input_masks = create_attention_mask(input_ids)

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(input_masks)

    # Get all the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    input_ids = input_ids.to(torch.long)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    sentence_features = last_hidden_states[0][:, 0, :].detach().numpy()

    # get the embedding sentence data shape = (number of sentence, BERT hidden layer)

    plt.clf()

    array_similarity = squareform(pdist(sentence_features, metric='euclidean'))
    sns.heatmap(array_similarity)
    plt.title('visualizing sentence semantic similarity')

    plt.savefig('static/plot1.png')
    plot1 = 'static/plot1.png'

    pca = PCA(n_components=2)
    pca.fit(sentence_features)

    pca_sentence_features = pca.transform(sentence_features)
    plt.figure(figsize=(10, 10))
    for i in range(len(pca_sentence_features)):
        plt.scatter(pca_sentence_features[i, 0], pca_sentence_features[i, 1])
        plt.annotate('sentence ' + str(i), (pca_sentence_features[i, 0], pca_sentence_features[i, 1]))
    plt.title('2D PCA projection of embedded sentences from CamemBERT')

    plt.savefig('static/plot2.png')
    plot2 = 'static/plot2.png'

    def find_cluster(hidden_array, sentence_list, n_topic=1, n_sentence_per_topic=3, resort_sentence=False):
        kmeans = KMeans(n_clusters=n_topic, random_state=0).fit(hidden_array)
        cluster_center = kmeans.cluster_centers_
        nbrs = NearestNeighbors(n_neighbors=n_sentence_per_topic, algorithm='brute').fit(hidden_array)
        distances, indices = nbrs.kneighbors(cluster_center.reshape(n_topic, -1))
        if not resort_sentence:
            indices = np.sort(indices)
        topic_answer = []
        for i in range(len(indices)):
            topic_i = []
            for j in indices[i]:
                topic_i.append(sentence_list[j])
            topic_answer.append(topic_i)
        return topic_answer

    test_answer = find_cluster(sentence_features, paragraph_split, n_topic=2, n_sentence_per_topic=3,
                               resort_sentence=True)

    def split_tokenize_formatting(paragraph, max_len=150, n_topic=1, n_sentence_per_topic=3, resort_sentence=False):
        paragraph_split = sent_tokenize(paragraph)
        input_tokens = []
        for i in paragraph_split:
            input_tokens.append(tokenizer.encode(i, add_special_tokens=True))
        temp = []
        for i in input_tokens:
            temp.append(len(i))
        if np.max(temp) > max_len:
            raise ValueError('sentence longer than the max_len')

        input_ids = pad_sequences(input_tokens, maxlen=max_len, dtype="long", value=0, truncating="post",
                                  padding="post")

        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]  # create a list of 0 and 1.
            attention_masks.append(att_mask)  # basically attention_masks is a list of list

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_masks)
        input_ids = input_ids.to(torch.long)
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        sentence_features = last_hidden_states[0][:, 0, :].detach().numpy()

        kmeans = KMeans(n_clusters=n_topic, random_state=0).fit(sentence_features)
        cluster_center = kmeans.cluster_centers_
        nbrs = NearestNeighbors(n_neighbors=n_sentence_per_topic, algorithm='brute').fit(sentence_features)
        distances, indices = nbrs.kneighbors(cluster_center.reshape(n_topic, -1))
        if not resort_sentence:
            indices = np.sort(indices)
        topic_answer = []
        for i in range(len(indices)):
            topic_i = []
            for j in indices[i]:
                topic_i.append(paragraph_split[j])
            topic_answer.append(topic_i)

        return topic_answer

    topic1 = split_tokenize_formatting(paragraph, max_len=maxLength, n_topic=n_topic, n_sentence_per_topic=n_sentence,
                                       resort_sentence=False)

    return topic1, plot1, plot2
