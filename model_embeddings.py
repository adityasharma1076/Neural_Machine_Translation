#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
from vocab import Vocab, VocabEntry# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
        self.embed_size = embed_size
        ### YOUR CODE HERE for part 1j
        self.max_word_length=21
        self.k_size=5
        self.output_embed_size = embed_size
        char_embed_size = 50
        
        pad_token_idx = vocab.char2id['<pad>']
        self.char_embeddings = nn.Embedding(len(vocab.char2id),char_embed_size,pad_token_idx)
        self.cnn=CNN(char_embed_size,self.output_embed_size,self.k_size,self.max_word_length)
        self.highway = Highway(self.output_embed_size)
        self.droput= nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        batch_embeddings = []

        for padded_sentence in input:
            char_embeddings = self.char_embeddings(padded_sentence)
            embeddings = torch.transpose(char_embeddings,dim0=1,dim1=2)
            cnn_output = self.cnn(embeddings)
            highway_output = self.highway(cnn_output)
            dropout_output = self.droput(highway_output)
            batch_embeddings.append(dropout_output)
        batch_embeddings = torch.stack(batch_embeddings)
        return batch_embeddings
        ### END YOUR CODE
# ## Test
# if __name__=="__main__": 
#     vocab = Vocab.load('vocab.json')
    
#     sentence_length = 10
#     max_word_length = 21
#     BATCH_SIZE = 15
#     model = ModelEmbeddings(embed_size=256,vocab=vocab.src)
#     inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
#     model.forward(inpt)