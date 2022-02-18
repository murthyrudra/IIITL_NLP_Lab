#!/usr/bin/env python3

from collections import Counter
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from q0 import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    wsd_token = sentence[word_index]
    wsd_lemma = wsd_token.lemma
    senses = list(wn.synsets(wsd_lemma))
    return(senses[0])


def lesk(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Simplified Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0
    context = [wsd_token.wordform for wsd_token in sentence]
    # context = stop_tokenize(' '.join(context)) # Remove stopwords and punctuation from context

    for sense in list(wn.synsets(sentence[word_index].lemma)):
        signature = get_signature(sense)
        score = overlap(signature, context)
        if score > best_score:
            best_sense = sense
            best_score = score

    return best_sense

def get_signature(sense: Synset) -> List[str]:
    '''Get the bag of words in the definition and examples of sense.
    '''
    signature = stop_tokenize(sense.definition())
    for example in sense.examples():
        signature += stop_tokenize(example) 
    # Remove target word from the bag 
    # sense_wordform = sense.name().split('.')[0]
    # signature = [w for w in signature if w != sense_wordform]
    return signature

def overlap(signature: List[str], context: List[str]) -> int:
    '''Calculates the number of overlapping word tokens for the simplified Lesk algorithm.
    '''
    # MARK: lowercased version
    # signature = [w.lower() for w in signature]
    # context = [w.lower() for w in context]

    cardinality = 0
    word_set = set(signature + context) # Set of all words in two bags
    counter_sig = Counter(signature)
    counter_ctxt = Counter(context)

    for word in word_set: 
        count_sig = counter_sig[word]
        count_ctxt = counter_ctxt[word]
        # 0 count means not in Counter
        if count_sig != 0 and count_ctxt != 0:
            cardinality += min(count_sig, count_ctxt)

    return cardinality

def lesk_ext(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0
    context = [wsd_token.wordform for wsd_token in sentence]

    for sense in list(wn.synsets(sentence[word_index].lemma)):
        signature = get_signature_ext(sense)
        score = overlap(signature, context)
        if score > best_score:
            best_sense = sense
            best_score = score

    return best_sense

def get_signature_ext(sense: Synset) -> List[str]:
    '''Get the bag of words in the definition and examples of sense
        plus definitions and examples of sense's hyponyms, holonyms, and meronyms.
    '''
    signature = get_signature(sense)

    for synset in list(sense.hyponyms())+list(sense.member_holonyms())+list(sense.substance_holonyms())+ \
                list(sense.part_holonyms())+list(sense.member_meronyms())+list(sense.substance_meronyms())+ \
                list(sense.part_meronyms()):

        signature += get_signature(synset)

    return signature

def lesk_cos(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0.0
    context = [wsd_token.wordform for wsd_token in sentence]

    for sense in list(wn.synsets(sentence[word_index].lemma)):
        signature = get_signature_ext(sense)
        score = cos_sim(signature, context)
        if score > best_score:
            best_sense = sense
            best_score = score

    return best_sense

def cos_sim(context: List[str], signature: List[str], one_sided=False) -> float:
    '''Calculates the cosine similarity of two bags of words.
    '''
    # MARK: lowercased version
    # signature = [w.lower() for w in signature]
    # context = [w.lower() for w in context]

    cos_score = 0.0
    word_set = sorted(set(signature + context)) # Set of all words in two bags sorted alphabatically 
    counter_sig = Counter(signature)
    counter_ctxt = Counter(context)
    vector_sig, vector_ctxt = [], []

    for word in word_set: 
        count_sig = counter_sig[word]
        count_ctxt = counter_ctxt[word]
        # Don't append this count if we are doign one-sided and it's not in the context.
        if one_sided and count_ctxt == 0:
            continue
        else:
            vector_sig.append(float(count_sig))
            vector_ctxt.append(float(count_ctxt))
    # Return 0 when denominator is 0
    if norm(vector_sig) == 0 or norm(vector_ctxt) == 0:
        cos_score = 0.0
    else:
        cos_score = np.dot(vector_sig, vector_ctxt)/(norm(vector_sig)*norm(vector_ctxt))
    return cos_score


def lesk_cos_onesided(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using one-sided cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0.0
    context = [wsd_token.wordform for wsd_token in sentence]

    for sense in list(wn.synsets(sentence[word_index].lemma)):
        signature = get_signature_ext(sense)
        score = cos_sim(signature, context, one_sided=True)
        if score > best_score:
            best_sense = sense
            best_score = score

    return best_sense


def lesk_w2v(sentence: Sequence[WSDToken], word_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    To look up the vector for a *single word*, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    But some wordforms are actually multi-word expressions and contain spaces.
    word2vec can handle multi-word expressions, but uses the underscore
    character to separate words rather than spaces. So, to look up a string
    that has a space in it, use the following rules:
    * If the string has a space in it, replace the space characters with
      underscore characters and then follow the above steps on the new string
      (i.e., try the string as-is, then the lower-cased version if that
      fails), but do not return the zero vector if the lookup fails.
    * If the version with underscores doesn't yield anything, split the
      string into multiple words according to the spaces and look each word
      up individually according to the rules in the above paragraph (i.e.,
      as-is, lower-cased, then zero). Take the mean of the vectors for each
      word and return that.
    Recursion will make for more compact code for these.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0.0
    
    context = [wsd_token.wordform for wsd_token in sentence]
    # MARK: lowercased version
    # context = [w.lower() for w in context]

    context = set(context) # count each word once only
    context_vec = np.empty(shape=(0, len(word2vec[0]))) # Make sure the size is same as word2vec
    
    for word in context:
        if '_' in word:
            word = word.replace('_', ' ')

        word_vec = convert_word_to_vector(word, vocab, word2vec)
        word_vec = np.reshape(word_vec, (1, -1))

        np.concatenate((context_vec, word_vec), axis=0)
    
    # Take mean of all word vectors to construct context
    # if context_vec.size == 0:
    #     context = np.zeros_like(context_vec)
    # else:
    context = np.mean(context_vec, axis=0)

    for sense in list(wn.synsets(sentence[word_index].lemma)):
        signature = get_signature_ext(sense)
        # MARK: lowercased version
        # signature = [w.lower() for w in signature]

        signature = set(signature)
        signature_vec = np.empty(shape=(0, len(word2vec[0])))

        for word in signature:
            if '_' in word:
                word = word.replace('_', ' ')

            word_vec = convert_word_to_vector(word, vocab, word2vec)
            word_vec = np.reshape(word_vec, (1, -1))

            np.concatenate((signature_vec, word_vec), axis=0)

        signature = np.mean(signature_vec, axis=0)

        score = np.dot(context, signature)/(norm(context)*norm(signature))
        if score > best_score:
            best_sense = sense
            best_score = score
    # print('Ambigious word:', sentence[word_index].wordform)
    # print('Context', context)
    # print('Signature', signature)
    # print('Best sense:', best_sense)
    return best_sense

def convert_word_to_vector(word: str, vocab: Mapping[str, int], word2vec: np.ndarray) -> np.ndarray:
    if ' ' in word:
        word_underscore = word.replace(' ', '_')

        word_vec = convert_word_to_vector(word_underscore, vocab, word2vec)

        if word_vec is not None: # Underscored version is found in word2vec
            return word_vec
        else: # Underscored version is not found, try splitting it up on spaces
            all_vec = np.empty((0, len(word2vec[0])))

            for split_word in word.split():
                np.concatenate((all_vec, np.reshape((convert_word_to_vector(split_word, vocab, word2vec)), (1, -1))))

            return np.mean(all_vec, axis=0) # Compute mean for each word

    else: # Regular words with no spaces
        if vocab.get(word) is not None: # Word in word2vec
            return word2vec[vocab[word]]
        elif vocab.get(word.lower()) is not None:
            return word2vec[vocab[word.lower()]]
        elif '_' in word:
            return None # Signal multi-word that underscored version is not found
        else:
            return np.zeros_like(word2vec[0])




if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()

    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos, lesk_cos_onesided]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
