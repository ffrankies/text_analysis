#!/usr/bin/python3
'''
Analyzes the readability of some texts.

CIS 365 - Artificial Intelligence - Project 2
Frank Wanye
Kellin McAvoy
Andrew Prins
'''

import nltk
import pyphen

def num_syllables(text):
    '''
    Finds the number of syllables in a given text.

    Params:
    - text (str): The word-tokenized text to be analyzed

    Returns:
    - num_syllables (int): The number of syllables in the given text
    '''
    dictionary = pyphen.Pyphen(lang='en_US')
    syllables = list()
    words = [word for word in text if word not in ['', ',', '.', '!', '?', ':', ';', '[', ']', '(', ')', '$', '@', '%']]
    for word in words:
        syllable_list = dictionary.inserted(word).split('-')
        syllables.extend(syllable_list)
    syllables = [syllable for syllable in syllables if len(syllable) > 0]
    print('Syllables: ', syllables)
    print('Num syllables: ', len(syllables))
    return len(syllables)
# End of num_syllables()

def num_words(text):
    '''
    Finds the number of words and syllables in a given text.

    Params:
    - text (str): The text to be analyzed

    Returns:
    - num_words (int): The number of words in the given text
    - num_syllables (int): The number of syllables in the given text
    '''
    words = nltk.tokenize.word_tokenize(text)
    syllables = num_syllables(words)
    print('Num words: ', len(words))
    return len(words), syllables
# End of num_words()

def num_sentences(text):
    '''
    Finds the number of sentences in a given text.

    Params:
    - text (str): The text to be analyzed

    Returns:
    - num_sentences (int): The number of sentences in the given text
    '''
    sentences = nltk.tokenize.sent_tokenize(text)
    print('Num sentences: ', len(sentences))
    return len(sentences)
# End of num_sentences()

def readability_score(text):
    '''
    Calculates the readability score for the given text.

    Params:
    - text (str): The text to be analyzed

    Returns:
    - score (float): The readability score for the text
    '''
    # syllables = num_syllables(text)
    words, syllables = num_words(text)
    sentences = num_sentences(text)
    score = 206.835
    score -= 1.015 * (words / sentences)
    score -= 84.6 * (syllables / words)
    return score
# End of readability_score()

if __name__ == "__main__":
    print(readability_score('Hello World!'))
    print(readability_score('Join the Dark Side, we have home-made cookies.'))
    print(readability_score('Once, into a quiet village, without haste and without heed '
                            'in the golden prime of morning, strayed the poet\'s winged steed. '
                            'It was autumn, and incessant, piped the quails from shocks and sheaves, '
                            'and, like living coals, the apples, burned among the withering leaves.'))