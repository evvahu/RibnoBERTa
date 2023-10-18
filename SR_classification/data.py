import numpy as np
import torch
from torch.utils.data import Dataset
from SR_classification.embedding_extractor import EmbeddingLoader
from sklearn import preprocessing


def create_label_encoder(all_labels):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(all_labels)
    return label_encoder

class DataProvider(Dataset):

    def __init__(self, sentences, targets,ids, labels,label_encoder, modelpath,batch_size, max_len, lower_case, max_pooling):

        """
        :param sentences: [List] A list of sentences that contain target word. 
        :param targets: [List] A list of target words
        :param labels: [List] A list of gold labels for words 
        :param label_encoder: [LabelEncoder] A LabelEncoder initialised with all labels
        :param modelpath: [String] The Transformer model path to be used for extracting contextualized word embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether the tokenizer should lower case words or not
        :param max_pooling: [boolean] Whether or not max pooling should be applied when extracting contextualized word embeddings 
        """

        assert len(sentences) == len(targets) & len(sentences) == len(labels), "length of sentences, targets and labels have to be equal"
        self.feature_extractor = EmbeddingLoader(modelpath=modelpath, max_len=max_len, batch_size=batch_size, max_pooling=max_pooling, lower_case=lower_case, device='cpu')
        self.sentences = sentences
        self.targets = targets
        self.ids = ids
        self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(labels)
        self.samples = self.populate_samples()

    def populate_samples(self):
        if len(self.ids) > 0:
            embeddings = self.feature_extractor.get_embeddings(sentences=self.sentences, target_words=self.targets, target_ids = self.ids)
        else:
            embeddings = self.feature_extractor.get_embeddings(sentences=self.sentences, target_words=self.targets)
        return [{"words": embeddings[i], "l": self.labels[i]} for i in
                range(len(self.labels))]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.samples[idx]
