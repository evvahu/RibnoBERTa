import torch
from transformers import AutoModel, AutoTokenizer, RobertaTokenizer, RobertaModel
import numpy as np 
import progressbar as pb
import sys
sys.path.append('..')
from SR_classification.SR_utils import format_indices

class EmbeddingLoader:
    def __init__(self, modelpath, max_len, batch_size, max_pooling=False, device='cpu', lower_case=True):
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.max_pooling = max_pooling
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, add_prefix_space=True, do_lower_case=lower_case)
        self.model = RobertaModel.from_pretrained(modelpath, output_hidden_states=True,)
        self.model.to(self.device)
        self.oov = self.tokenizer.convert_tokens_to_ids('[oov]')


    def convert_sentence_to_indices(self, sentence):
        """
        :param sentence: A String representing a sentence, without special tokens
        :return: A dictionary containing the following:
            input ids: list[int] (the word piece ids of the given sentence)
            token_type_ids: list[int] (the token type ids, in this case a list of 0, because we do not consider a
            pair of sequences
            attention_mask: list[int] (this list marks the real word pieces with 1 and the padded with 0)
        """
        return self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_len, 
                                          pad_to_max_length=True)
    
    def get_embeddings(self, sentences, target_words, target_ids=[]):
        sentence_batches = self.get_sentence_batches(sentences)
        target_word_batches = self.get_sentence_batches(target_words)
        target_id_batches = self.get_sentence_batches(target_ids)
        contextualized_embeddings = []
        for i in pb.progressbar(range(0, len(sentence_batches))):

            current_sentences = sentence_batches[i]
            current_target_words = target_word_batches[i]
            current_target_ids = target_id_batches[i]
            #batch_input_ids, batch_token_type_ids, batch_attention_mask = self.convert_sentence_batch_to_indices(current_sentences)
            batch_input_ids, batch_attention_mask = self.convert_sentence_batch_to_indices(current_sentences)
            # shape last_hidden_states : batch_size x max_len x embedding_dim
            last_hidden_states, all_layers = self.get_vectors(batch_input_ids=batch_input_ids,
                                                                   batch_attention_mask=batch_attention_mask) #,
                                                                  # batch_token_type_ids=batch_token_type_ids)
            if self.max_pooling:
                mean_layers = self.get_mean_layer_pooling(all_layers, 0, 11)
            else:
                mean_layers = all_layers[-1] 
            for i in range(len(current_sentences)):
                if type(current_sentences) != list:
                    current_target_ids = current_target_ids.tolist()
                    current_sentences = current_sentences.tolist()

                target_word_indices = self.word_id_in_sent_ids(current_sentences[i], current_target_ids[i]) #here one could check that target_ids = target_words
                token_embeddings = mean_layers[i]
                contextualized_emb = self.get_single_word_embedding(token_embeddings=token_embeddings,
                                                                    target_word_indices=target_word_indices)
                contextualized_embeddings.append(contextualized_emb.to("cpu"))
        return torch.stack(contextualized_embeddings)
            

    def get_sentence_batches(self, sentences):
        """
        This method splits a list of sentences into batches for efficient processing.
        :param sentences: [String] A list of sentences
        :return: A list of lists, each list except the last element are of length batchsize.
        """
        sentence_batches = []

        for i in range(0, len(sentences), self.batch_size):
            if i + self.batch_size - 1 > len(sentences):
                sentence_batches.append(sentences[i:])
            else:
                sentence_batches.append(sentences[i:i + self.batch_size])
        return sentence_batches

    def convert_sentence_batch_to_indices(self, sentences):
        """
        Having a list of sentences as input this method returns the corresponding input_ids, token_type_ids and
        attention_masks for each sentence.
        :param sentences: a list of Strings.
        :return: three matrices, each vector corresponds to a certain type of id of a specific sentence
        """
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_mask = []
        for sentence in sentences:
            converted = self.convert_sentence_to_indices(sentence=sentence)
            batch_input_ids.append(converted['input_ids'])
            #batch_token_type_ids.append(converted['token_type_ids'])
            batch_attention_mask.append(converted['attention_mask'])
        #return batch_input_ids, batch_token_type_ids, batch_attention_mask
       
        return batch_input_ids, batch_attention_mask 

    def word_id_in_sent_ids(self, sentence, target_ids):
        """
        Given a sentence and a target indices, this method returns the indices of the tokenized word within the sentence,
        when special tokens were added. So for the sentence "This is a sentence." we can assume that the Bert Tokenizer
        will tokenize it the following way: ['[CLS]', 'Th', '##is', 'is', 'a', 'sen', '##ten', '##ce', '.', '[SEP]]'
        Then for the target word 'sentence' we will retrieve a list with the corresponding indices - list([5, 6, 7])
        :param sentence: A String representing the sentence without special Tokens
        :param target_ids: A list with indices of words in the sentence
        :return: A list corresponding to the sequence of indices that match the target word in the given sentence
        """
        sentence = sentence.replace('  ', ' ')
        split_sentence = sentence.strip().split(' ')

        target_ids = format_indices(target_ids)
        assert max(target_ids) < len(split_sentence), "target id cannot be higher than nr of sentence ids"
        target_words = []
        for id in target_ids:
            target_words.append(split_sentence[id])
        target_words = ' '.join(target_words).strip()
        sentence_indices = self.tokenizer.encode(sentence, add_special_tokens=True) 
        word_indices = self.tokenizer.encode(target_words, add_special_tokens=False)

        for i in range(len(sentence_indices)):

            if sentence_indices[i] == word_indices[0] and sentence_indices[i:i + len(word_indices)] == word_indices:
                startindex = i
                break
        return np.arange(startindex, startindex+len(word_indices)) 


    def word_id_in_sent_words(self, sentence, target_word):
        """
        Given a sentence and a target word, this method returns the indices of the tokenized word within the sentence,
        when special tokens were added. So for the sentence "This is a sentence." we can assume that the Bert Tokenizer
        will tokenize it the following way: ['[CLS]', 'Th', '##is', 'is', 'a', 'sen', '##ten', '##ce', '.', '[SEP]]'
        Then for the target word 'sentence' we will retrieve a list with the corresponding indices - list([5, 6, 7])
        :param sentence: A String representing the sentence without special Tokens
        :param target_word: A String representing the target word
        :return: A list corresponding to the sequence of indices that match the target word in the given sentence
        """
        assert target_word in sentence, "target word must be contained in the context sentence!"
        sentence_indices = self.tokenizer.encode(sentence, add_special_tokens=True)
        word_indices = self.tokenizer.encode(target_word, add_special_tokens=False)
        startindex = 0
        for i in range(len(sentence_indices)):
            if sentence_indices[i] == word_indices[0] and sentence_indices[i:i + len(word_indices)] == word_indices:
                startindex = i
                break
        
        return np.arange(startindex, startindex+len(word_indices))

    @staticmethod
    def get_single_word_embedding(token_embeddings, target_word_indices):
        """
        For a list of token embeddings, extract the centroid of all word piece embeddings that belong to a target word.
        :param token_embeddings: A list of word piece embeddings that are part of a sentence.
        :param target_word_indices: A list of target word piece indices, that mark the indices of the target word in the
        list of token embeddings.
        :return: The centroid of the word piece embeddings (to do: other pooling options needed ? )
        """
        if len(target_word_indices) == 1:
            return token_embeddings[target_word_indices[0]]
        else:
            target_word_tokens = token_embeddings[
                                 target_word_indices[0]:target_word_indices[len(target_word_indices) - 1]]
            return target_word_tokens.sum(0) / target_word_tokens.shape[0]

    def get_vectors(self, batch_input_ids, batch_attention_mask): #,batch_token_type_ids):
        """
        For a batch of input_ids, attention_masks, type_token_ids, Bert computes the corresponding layers for each token
        for each sentence in the batch.
        :param batch_input_ids: batch of the word piece ids
        :param batch_attention_mask: batch of attention masks  (what word pieces are real and what are padded)
        :param batch_token_type_ids: batch of token_type_ids
        :return: last_hidden_states: the top layer for each token within each sentence.
                 all_layers: all 12 layers for each token within each sentence.
        """
        with torch.no_grad():
            #last_hidden_states, _, all_layers = self.model(input_ids=torch.tensor(batch_input_ids).to(self.device),
             #                                              attention_mask=torch.tensor(batch_attention_mask).to(self.device))# ,
                                                         #  token_type_ids=torch.tensor(batch_token_type_ids).to(self.device))
            output = self.model(input_ids=torch.tensor(batch_input_ids).to(self.device),attention_mask=torch.tensor(batch_attention_mask).to(self.device))
                                      
        return output.last_hidden_state, output.hidden_states #all_layers

    @staticmethod
    def get_mean_layer_pooling(layers, first_layer, last_layer):
        """
        Given a list of layers for a sentence, return one representation for the sentence by taking the mean of the
        specified layers.
        :param layers: The list of hidden states that is the output of Bert.
        :param first_layer: the first layer to include
        :param last_layer: the last layer to include
        :return: A torch tensor, representing the sentence vector (as the pooled mean of specified layers)
        """
        assert first_layer <= last_layer, "the lower layer must be lower than the upper layer"
        return torch.mean(torch.stack(layers[first_layer:last_layer]), dim=0)


    @staticmethod
    def get_mean_layer_pooling(layers, first_layer, last_layer):
        """
        Given a list of layers for a sentence, return one representation for the sentence by taking the mean of the
        specified layers.
        :param layers: The list of hidden states that is the output of Bert.
        :param first_layer: the first layer to include
        :param last_layer: the last layer to include
        :return: A torch tensor, representing the sentence vector (as the pooled mean of specified layers)
        """
        assert first_layer <= last_layer, "the lower layer must be lower than the upper layer"
        return torch.mean(torch.stack(layers[first_layer:last_layer]), dim=0)