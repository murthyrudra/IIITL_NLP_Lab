#!/usr/bin/env python3

import typing as T
from string import punctuation

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize


def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    syn_depths = []
    for synset in list(wn.all_synsets()):
        syn_depths.append((synset.max_depth(), synset))
    syn_depths.sort(reverse=True) # Deepest item is at [0] now

    (depth, deepest_syn) = syn_depths[0]
    print('Deepest Synset:', deepest_syn.name())
    print('Max Depth:', depth)
    print('Depths of different paths to root:', [len(path)-1 for path in deepest_syn.hypernym_paths()])


def superdefn(s: str) -> T.List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up...)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        s (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    syn = wn.synset(s)
    super_def = []
    # Get definitions
    super_def += word_tokenize(syn.definition())
    for hyper in syn.hypernyms(): super_def += word_tokenize(hyper.definition())
    for hypo in syn.hyponyms(): super_def += word_tokenize(hypo.definition())
    return super_def


def stop_tokenize(s: str) -> T.List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """
    text = word_tokenize(s)
    # Remove stopwords and punctuation 
    content = [w for w in text if w.lower() not in stopwords.words('english')]
    word_tokens = [t for t in content if t not in punctuation]
    return word_tokens

if __name__ == '__main__':
    import doctest
    doctest.testmod()
