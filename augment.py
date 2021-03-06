#!/usr/bin/python3
'''
Augments the difficulty of the text document by removing the least significant
words in a text and then replacing remaining words with the shortest synonym.

Git Repository: https://github.com/ffrankies/text_analysis

CIS 365 - Artificial Intelligence - Project 2
Frank Wanye
Kellin McAvoy
Andrew Prins
Troy Madsen
'''

import analyze
import pandas
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

#
# Paths to Data Files
#
AUGMENTED_DATA = './processed_data/news/augmented.pkl'

# Things to not count as words and/or syllables
WORD_EXCEPTIONS = ['', ',', '.', '!', '?', ':', ';', '[', ']', '(', ')', '$', '@', '%', '\'', '"', '`', '”', '“', '’']

'''
Augments the difficult of the specified text to make it easier to read

Params:
- text (str): The text to augment

Returns:
- augmented_text (str): The augmented version of the text
'''
def augment(text):
    # Printing the old text
    # print(text)

    # Augmenting the text

    # Creating key-value pairs for tokenized words
    words = word_tokenize(text)
    for index, word in enumerate(words):
        words[index] = [index, word]

    # Stopwords to remove meaningless words
    stop_words = set(stopwords.words("english"))

    # Getting only the words we want to augment
    aug_words = [w for w in words if not w[1] in stop_words]

    # Replacing words with the best synonym
    for aw in aug_words:
        # Track shortest syllables and word
        shortest = analyze.syllables_in_word(aw[1])
        shortest_word = aw[1]

        # For each synonym
        for syn in wordnet.synsets(aw[1]):
            # For each lemma
            for l in syn.lemmas():
                # Update shortest word
                syllables = analyze.syllables_in_word(l.name())
                if syllables < shortest:
                    # print(aw[1] + ' ' + l.name())
                    shortest = syllables
                    shortest_word = l.name()

        # Replace word with the shortest alternative
        aw[1] = shortest_word

    # Inserting augmented words into tokenized word list
    for w in aug_words:
        words[w[0]] = w

    # Replacing key-value pairs with simply the str of each
    words = [w[1] for w in words]

    # Inserting spaces into the text
    for i in range(len(words) - 1, -1, -1):
        if not words[i] in WORD_EXCEPTIONS:
            words.insert(i, ' ')

    # Flatten words in a single str
    augmented_text = ''.join(words)

    # Printing the augemnted text
    # print(augmented_text)

    return augmented_text
# End of augment()

def process_news_data():
    '''
    Combines all the news data into one pandas DataFrame, calculates the readability_score for all articles, and
    appends that to the DataFrame.

    Returns
    - news_data (pandas.DataFrame): The data frame containing processed news article data
    '''
    news_data = analyze.read_news_data()
    print('=====Processing News Data=====')
    scores = list()
    aug_scores = list()
    num_errors = 0
    for index, content in enumerate(news_data['content']):
        score = analyze.readability_score(content)
        content = augment(content)
        new_score = analyze.readability_score(content)
        if score == -1:
            num_errors += 1
        scores.append(score)
        aug_scores.append(new_score)
        if index % 1000 == 0:
            print('Processed %d articles' % index)

    print("Articles with a faulty score: %d/%d" % (num_errors, len(scores)))
    news_data['score'] = scores + [scores[0]] * (len(news_data) - len(scores))
    news_data['aug_scores'] = aug_scores + [aug_scores[0]] * (len(news_data) - len(aug_scores))
    news_data = news_data.loc[news_data['score'] > -1] # Drop all rows where the score is wrong
    news_data = news_data.drop(columns=['content'])
    print(news_data.head())
    analyze.save_processed_data(news_data, AUGMENTED_DATA)
    return news_data
# End of process_news_data()

if __name__ == "__main__":
    process_news_data()
