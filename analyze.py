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
import dill
import pathlib
import argparse
import matplotlib
matplotlib.use('Agg') # Otherwise it crashes
import matplotlib.pyplot as plt
import seaborn as sns

#
# Paths to Data Files
#
NEWS_DATA = './processed_data/news/articles.pkl'

# Things to not count as words and/or syllables
WORD_EXCEPTIONS = ['', ',', '.', '!', '?', ':', ';', '[', ']', '(', ')', '$', '@', '%', '\'', '"', '`', '”', '“']

#
# The dictionaries for counting syllables
#
CMUDICT = nltk.corpus.cmudict.dict()
PYPHENDICT = pyphen.Pyphen(lang='en_US')

def syllables_in_word(word):
    '''
    Returns the number of syllables in a word. If the word is in nltk's cmudict dictionary. This dictionary contains
    pronunciation guide strings for words in the Carnegie Mellon University dictionary. E.g.: the entry for the word
    'syllable' is ['S', 'IH1', 'L', 'AH0', 'B', 'AH0', 'L']. The vowels that are stressed in pronunctiation (which
    correspond to syllables in a word with very few exceptions, if any), are denoted by having a digit at the end of
    the entry (eg, for 'syllable', the stresed vowel sounds are 'IH1', 'AH0' and 'AH0'). The number of syllabels is
    then equal to the number of strings in the cmudict entry for a word that end in digits.

    If a word is not present in the cmudict, then we resort to pyphen, which hyphenates the word. It is then split on
    the hyphen character, and the length of the resulting list is used as an estimate of the number of syllables in
    the word.

    Params:
    - word (str): The word for which we want to count syllables

    Returns:
    - num_syllables (int): The number of syllables in the word
    '''
    word = word.lower()
    if word in CMUDICT:
        cmudict_entry = CMUDICT[word][0]
        cmudict_entry = map(lambda part : 1 if part[-1].isdigit() else 0, cmudict_entry)
        cmudict_entry = list(cmudict_entry)
        num_syllables = sum(cmudict_entry)
    else:
        syllable_list = PYPHENDICT.inserted(word).split('-')
        num_syllables = len(syllable_list)
    return num_syllables
# End of syllables_in_word()

def num_syllables(text):
    '''
    Finds the number of syllables in a given text.

    Params:
    - text (str): The word-tokenized text to be analyzed

    Returns:
    - num_syllables (int): The number of syllables in the given text
    '''
    num_syllables = 0
    for word in text:
        num_syllables += syllables_in_word(word)
    return num_syllables
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
    words = filter(lambda word : word not in WORD_EXCEPTIONS, words)
    words = list(words)
    syllables = num_syllables(words)
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
    return len(sentences)
# End of num_sentences()

def readability_score(text):
    '''
    Calculates the Flesch-Kincaid Reading Easy Score for the given text. If the readability score is unable
    to be calculated, returns 1000

    Params:
    - text (str): The text to be analyzed

    Returns:
    - score (float): The readability score for the text
    '''
    words, syllables = num_words(text)
    sentences = num_sentences(text)
    if sentences == 0 or words == 0: # Prevent division by 0
        return 1000
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
    num_errors = 0
    for index, content in enumerate(news_data['content']):
        score = readability_score(content)
        if score > 200:
            num_errors += 1
        scores.append(score)
        if index % 1000 == 0:
            print('Processed %d articles' % index)
    print("Articles with a faulty score: %d/%d" % (num_errors, len(scores)))
    news_data['score'] = scores
    news_data = news_data.loc[news_data['score'] > 200] # Drop all rows where the score is wrong
    news_data = news_data.drop(columns=['content'])
    print(news_data.head())
    save_processed_data(news_data, NEWS_DATA)
    return news_data
# End of process_news_data()

def save_processed_data(data, filePath):
    '''
    Saves the processed data in file at the given path using the dill module.

    Params:
    - data (pandas.DataFrame): The processed data to save
    - filePath (str): The path in which to save the data
    '''
    pathlib.Path(filePath).parent.mkdir(parents=True, exist_ok=True)
    with open(filePath, 'wb') as dataFile:
        dill.dump(data, dataFile)
# End of save_processed_data()

def load_processed_data(filePath):
    '''
    Loads processed data from the given file.

    Params:
    - filePath (str): The path to the file containing processed data

    Returns:
    - data (pandas.DataFrame): The pandas data frame containing processed data
    '''
    with open(filePath, 'rb') as dataFile:
        data = dill.load(dataFile)
    print(data.head())
    return data
# End of load_processed_data()

def analyze_news_data(news_data):
    '''
    Creates multiple plots showing different aspects of news data.

    Params:
    - news_data (pandas.DataFrame): The processed news data
    '''
    # Scores distribution
    plot = sns.distplot(news_data['score'], bins=150)
    plot.set_title('Distribution of Flesch-Kincaid Reading Ease Scores for News Articles')
    plot.set(xlim=(0,100), yticks=[], xlabel='Flesch-Kincaid Reading Ease Score')
    plt.tight_layout()
    save_plot(plot, './analysis/news/distribution.png')
    # Scores by publication
    plot = sns.boxplot(y='publication', x='score', data=news_data)
    plot.set_title('Comparing Flesch-Kincaid Reading Ease Score\nDistributions Across Publications')
    plot.set(xlim=(0,100), xlabel='Flesch-Kincaid Reading Ease Score', ylabel='')
    plt.tight_layout()
    save_plot(plot, './analysis/news/publication_boxplot.png')
    publication_means = news_data.groupby(['publication'], as_index=False).mean()
    print(publication_means.head())
    plot = sns.barplot(y='publication', x='score', data=publication_means)
    plot.set_title('Comparing Flesch-Kincaid Reading Ease Scores\nfor Different Publications')
    plot.set(xlim=(0,70), xlabel='Average Flesch-Kincaid Reading Ease Score', ylabel='')
    plt.tight_layout()
    save_plot(plot, './analysis/news/publication_comparison.png')
    # Scores by year
    year_means = news_data.groupby(['year'], as_index=False).mean()
    year_means['year'] = year_means['year'].astype(int)
    year_means['year'] = year_means['year'].astype(str)
    print(year_means.head())
    plot = sns.pointplot(x='year', y='score', data=year_means)
    plot.set_title('Comparing Flesch-Kincaid Reading Ease Scores\nfor Different Years')
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plot.set(ylim=(20,80), ylabel='Average Flesch-Kincaid Reading Ease Score', xlabel='')
    plt.tight_layout()
    save_plot(plot, './analysis/news/year_comparison.png')
# End of analyze_news_data()

def save_plot(plot, filePath):
    '''
    Saves the plot figure into the given filePath.

    Params:
    - plot (plt.Axes): The axes containing the plot to save
    - filePath (str): The path into which to save the plot
    '''
    pathlib.Path(filePath).parent.mkdir(parents=True, exist_ok=True)
    fig = plot.get_figure()
    fig.savefig(filePath)
    plt.cla()
# End of save_plot()

def parse_arguments():
    '''
    Parses the command-line arguments to the script.

    Returns:
    - args (argparse.Namespace): The namespace object containing the parsed arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--news', '-n', action='store_true', help='Process news data')
    args = parser.parse_args()
    return args
# End of parse_arguments()

if __name__ == "__main__":
    args = parse_arguments()
    if args.news:
        news_data = process_news_data()
    else:
        news_data = load_processed_data(NEWS_DATA)
    sns.set() # Use seaborn's plot styling
    analyze_news_data(news_data)
