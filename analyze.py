#!/usr/bin/python3
'''
Analyzes the readability of some texts.

CIS 365 - Artificial Intelligence - Project 2
Frank Wanye
Kellin McAvoy
Andrew Prins
Troy Madsen
'''

import nltk
import pyphen
import pandas

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
    # print('Syllables: ', syllables)
    # print('Num syllables: ', len(syllables))
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
    # print('Num words: ', len(words))
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
    # print('Num sentences: ', len(sentences))
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
    if sentences == 0 or words == 0:
        return -1
    score = 206.835
    score -= 1.015 * (words / sentences)
    score -= 84.6 * (syllables / words)
    return score
# End of readability_score()

def read_news_data():
    '''
    Reads in news article data, removes unneeded columns, and combines the three csv files into one pandas data frame.

    Returns:
    - news_data (pandas.DataFrame): The data frame containing processed news article data
    '''
    print('=====Reading News Data=====')
    columns_to_drop = ['id', 'title', 'author', 'date', 'month', 'url'] # Leaves publication, year and content
    data_frame_1 = pandas.read_csv('./data/news/articles1.csv')
    data_frame_1 = data_frame_1.drop(columns=columns_to_drop)
    data_frame_2 = pandas.read_csv('./data/news/articles2.csv')
    data_frame_2 = data_frame_2.drop(columns=columns_to_drop)
    data_frame_3 = pandas.read_csv('./data/news/articles3.csv')
    data_frame_3 = data_frame_3.drop(columns=columns_to_drop)
    news_data = pandas.concat([data_frame_1, data_frame_2, data_frame_3])
    return news_data
# End of read_news_data()

def process_news_data():
    '''
    Combines all the news data into one pandas DataFrame, calculates the readability_score for all articles, and 
    appends that to the DataFrame.

    Returns
    - news_data (pandas.DataFrame): The data frame containing processed news article data
    '''
    news_data = read_news_data()
    print('=====Processing News Data=====')
    scores = list()
    for index, content in enumerate(news_data['content']):
        scores.append(readability_score(content))
        if index % 1000 == 0:
            print('Processed %d articles' % index)
    news_data['score'] = scores
    news_data = news_data.drop(columns=['content'])
    print(news_data.head())
    return news_data
# End of process_news_data()

if __name__ == "__main__":
    news_data = process_news_data()
    print(readability_score('Hello World!'))
    print(readability_score('Join the Dark Side, we have home-made cookies.'))
    print(readability_score('Once, into a quiet village, without haste and without heed '
                            'in the golden prime of morning, strayed the poet\'s winged steed. '
                            'It was autumn, and incessant, piped the quails from shocks and sheaves, '
                            'and, like living coals, the apples, burned among the withering leaves.'))