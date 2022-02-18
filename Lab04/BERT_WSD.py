#!/usr/bin/env python3

import enum
import typing as T
from collections import defaultdict

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import torch
from torch import Tensor
from torch.linalg import norm

from tqdm import tqdm, trange

from q1 import mfs
from wsd import (batch_evaluate, load_bert, run_bert, load_eval, load_train,
                 WSDToken)


def gather_sense_vectors(corpus: T.List[T.List[WSDToken]],
                         batch_size: int = 32) -> T.Dict[str, Tensor]:
    """Gather sense vectors using BERT run over a corpus.

    As with A1, it is much more efficient to batch the sentences up than it is
    to do one sentence at a time, and you can further improve (~twice as fast)
    if you sort the corpus by sentence length first. We've therefore started
    this function out that way for you, but you may implement the code in this
    function however you like.

    The procedure for this function is as follows:
    * Use run_bert to run BERT on each batch
    * Go through all of the WSDsentence_token in the input batch. For each one, if the
      token has any synsets assigned to it (check WSDToken.synsets), then add
      the BERT output vector to a list of vectors for that sense (**not** for
      the token!).
    * Once this is done for all batches, then for each synset that was seen
      in the corpus, compute the mean of all vectors stored in its list.
    * That yields a single vector associated to each synset; return this as
      a dictionary.

    The run_bert function will handle tokenizing the batch for BERT, including
    padding the tokenized sentences so that each one has the same length, as
    well as converting it to a PyTorch tensor that lives on the GPU. It then
    runs BERT on it and returns the output vectors from the top layer.

    An important point: the tokenizer will produce more sentence_token than in the
    original input, because sometimes it will split one word into multiple
    pieces. BERT will then produce one vector per token. In order to
    produce a single vector for each *original* word token, so that you can
    then use that vector for its various synsets, you will need to align the
    output sentence_token back to the originals. You will then sometimes have multiple
    vectors for a single token in the input data; take the mean of these to
    yield a single vector per token. This vector can then be used like any
    other in the procedure described above.


    To provide the needed information to compute the token-word alignments,
    run_bert returns an offset mapping. For each token, the offset mapping
    provides substring indices, indicating the position of the token in the
    original word (or [0, 0] if the token doesn't correspond to any word in the
    original input, such as the [CLS], [SEP], and [PAD] sentence_token). You can
    inspect the returned values from run-bert in a debugger and/or try running
    the tokenizer on your own test inputs. Below are a couple examples, but
    keep in mind that these are provided purely for illustrative purposes
    and your actual code isn't to call the tokenizer directly itself.
        >>> from wsd import load_bert
        >>> load_bert()
        >>> from wsd import TOKENIZER as tknz
        >>> tknz('This is definitely a sentence.')
        {'input_ids': [101, 1188, 1110, 5397, 170, 5650, 119, 102],
         'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
         'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
        >>> out = tknz([['Multiple', ',', 'pre-tokenized', 'sentences', '!'], \
                        ['Much', 'wow', '!']], is_split_into_words=True, \
                        padding=True, return_offsets_mapping=True)
        >>> out.sentence_token(0)
        ['[CLS]', 'Multiple', ',', 'pre', '-', 'token', '##ized', 'sentences',
         '!', '[SEP]']
        >>> out.sentence_token(1)
        ['[CLS]', 'Much', 'w', '##ow', '!', '[SEP]', '[PAD]', '[PAD]', '[PAD]',
        '[PAD]']
        >>> out['offset_mapping']
        [[[0, 0], [0, 8], [0, 1], [0, 3], [3, 4], [4, 9], [9, 13], [0, 9],
         [0, 1], [0, 0]],
         [[0, 0], [0, 4], [0, 1], [1, 3], [0, 1], [0, 0], [0, 0], [0, 0],
         [0, 0], [0, 0]]]

    Args:
        corpus (list of list of WSDToken): The corpus to use.
        batch_size (int): The batch size to use.

    Returns:
        dict mapping synsets IDs to Tensor: A dictionary that can be used to
        retrieve the (PyTorch) vector for a given sense.
    """
    corpus = sorted(corpus, key=len)
    # The dictionary for sense-vector pairs
    word_dict = {}
    for batch_n in trange(0, len(corpus), batch_size, desc='gathering',
                          leave=False):
        batch = corpus[batch_n:batch_n + batch_size] # List[List[WSDToken]]
        
        # Use str format to run BERT
        batch_wordform = [[word.wordform for word in sentence] for sentence in batch]
        batch_token, batch_offset = run_bert(batch_wordform)

        batch_vector = []
        # Align the output sentence_token back to the originals
        for sentence_index, sentence_token in enumerate(batch_token):
            sentence_offset = batch_offset[sentence_index]

            sentence_vector = align_bert_tokens(sentence_token, sentence_offset) # List of Tensors
            batch_vector.append(sentence_vector)
        
        # Add sanity check for sizes of wordform and tensor form
        # if len(batch_wordform) != len(batch_vector) or len(batch_wordform[0]) != len(batch_vector[0]):
            # raise ValueError('Size does not match after BERT realignment')
        
        for sentence_index, sentence in enumerate(batch):
            for wsd_index, wsd_token in enumerate(sentence):
                wsd_senses = wsd_token.synsets
                if len(wsd_senses) > 0: # Has synsets assigned to it
                    for sense in wsd_senses:
                        reshaped_tensor = torch.reshape(batch_vector[sentence_index][wsd_index], (1, -1)) # Vector for this word
                        dict_value = word_dict.get(sense) # Check if in dictionary
                        if dict_value is None: 
                            # This sense is not in dictionary before, add it in
                            word_dict[sense] = reshaped_tensor
                        else:
                            word_dict[sense] = torch.cat((dict_value, reshaped_tensor), dim=0)

        # Take mean for each key-value pair
        for key in word_dict:
            word_dict[key] = torch.mean(word_dict[key], dim=0, keepdim=True)

    # for key in word_dict:
        # print('Dictionary key:', key)
    return word_dict
                
# Helper function for gather_sense_vectors
def align_bert_tokens(sentence_token: Tensor, sentence_offset: Tensor) -> T.List[Tensor]:
    ''' Align tokens generated by BERT to the original wordform
    '''
    sentence_vector = []
    word_vector = [] # Holder for a word's vectors
    for word_index, word_token in enumerate(sentence_token):

        offset = sentence_offset[word_index] # A tensor offset mapping in format of [start, end]

        if torch.equal(offset, torch.tensor([0, 0])) or offset[0].item() == 0:
            # New token, check if need to process the previous token
            if len(word_vector) > 0:
                # Take mean of the previous word's vectors
                word_tensor = torch.cat(([torch.reshape(t, (1, -1)) for t in word_vector]), dim=0)
                sentence_vector.append(torch.mean(word_tensor, dim=0, keepdim=True))
            word_vector = []

        if torch.equal(offset, torch.tensor([0, 0])): 
            continue 
        else: 
            # print('New word token or continued from last token')
            word_vector.append(word_token)
    return sentence_vector
        

def bert_1nn(batch: T.List[T.List[WSDToken]],
             indices: T.Iterable[T.Iterable[int]],
             sense_vectors: T.Mapping[str, Tensor]) -> T.List[T.List[Synset]]:
    """Find the best sense for specified words in a batch of sentences using
    the most cosine-similar sense vector.

    See the docstring for gather_sense_vectors above for examples of how to use
    BERT. You will need to run BERT on the input batch and associate a single
    vector for each input token in the same way. Once you've done this, you can
    compare the vector for the target word with the sense vectors for its
    possible senses, and then return the sense with the highest cosine
    similarity.

    In case none of the senses have vectors, return the most frequent sense
    (e.g., by just calling mfs(), which has been imported from q1 for you).

    **IMPORTANT**: When computing the cosine similarities and finding the sense
    vector with the highest one for a given target word, do not use any loops.
    Implement this aspect via matrix-vector multiplication and other PyTorch
    ops.

    Args:
        batch (list of list of WSDToken): The batch of sentences containing
            words to be disambiguated.
        indices (list of list of int): The indices of the target words in the
            batch sentences.
        sense_vectors: A dictionary mapping synset IDs to PyTorch vectors, as
            generated by gather_sense_vectors(...).

    Returns:
        predictions: The predictions of the correct sense for the given words.
    """
    batch_wordform = [[word.wordform for word in sentence] for sentence in batch]
    batch_token, batch_offset = run_bert(batch_wordform)

    best_sense_vec = [] # Best senses to return
    # Align the output sentence_token back to the originals
    for sentence_index, sentence_token in enumerate(batch_token):
        sentence_offset = batch_offset[sentence_index]

        sentence_vector = align_bert_tokens(sentence_token, sentence_offset) # List of Tensors
        
        # One sentence may have multiple indices to disambiguite
        sense_list = [] 
        for word_index in indices[sentence_index]:
            wsd_token = batch[sentence_index][word_index]
            word_vector = sentence_vector[word_index] # output vector from BERT

            best_score = 0.0
            best_sense = None

            # Find the best sense using cosine similarity
            for sense in list(wn.synsets(wsd_token.lemma)):
                # Convert Synset to str
                sense_str = sense.name() 
                # Check whether the sense is in the dictionary
                if sense_vectors.get(sense_str) is None:
                    # print('WordNet sense not in dictionary')
                    continue # Sense not in dictionary - ignore
                # Compute Cosine Similarity
                sense_vec_norm = sense_vectors[sense_str] / norm(sense_vectors[sense_str])
                word_vec_norm = word_vector / norm(word_vector)
                cos_score = torch.mm(sense_vec_norm, word_vec_norm.transpose(0,1))
                
                if cos_score > best_score:
                    best_sense = sense
                    best_score = cos_score

            if best_sense is None:
                best_sense = mfs(batch[sentence_index], word_index)

            sense_list.append(best_sense)

        best_sense_vec.append(sense_list)
    
    return best_sense_vec



if __name__ == '__main__':
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        tqdm.write(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        tqdm.write('Running on CPU.')

    with torch.no_grad():
        load_bert()
        train_data = load_train()
        eval_data = load_eval()

        sense_vecs = gather_sense_vectors(train_data)
        batch_evaluate(eval_data, bert_1nn, sense_vecs)
