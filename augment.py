#!/usr/bin/python3
'''
Creates a simpler and more complex version of a text.

CIS 365 - Artificial Intelligence - Project 2
Kellin McAvoy
Frank Wanye
Andrew Prins
Troy Madsen
'''

import copy
import nltk
import analyze
from nltk.corpus import wordnet

# Things to not count as words and/or syllables
WORD_EXCEPTIONS = ['', ',', '.', '!', '?', ':', ';', '[', ']', '(', ')', '$', '@', '%', '\'', '"', '`', '”', '“']

def speech_part(tag):

    '''
    Determines if part of speech is easily changable.

    Returns:
    - True if part of speech is verb, adjective, adverb or non-proper noun
    - False otherwise
    '''
    PARTS_OF_SPEECH = ['NN', 'VB', 'RR', 'JJ']
    POS_EXCEPTIONS = ['NNP', 'NNPS']
    for part in PARTS_OF_SPEECH:
        if tag.startswith(part) and tag not in POS_EXCEPTIONS:
            return True
    return False
# End of speech_part()

def sentence_join(list, delimiter=" "):

    '''
    Combines str elements in a list together, seperated by delimiter,
    but does not put spaces between punctuation.

    Returns:
    - String created by elements in list
    '''
    str = ""
    for item in list:
        if item in WORD_EXCEPTIONS:
            str = str[:-1]
        str += item + delimiter
    return str

def augment(text):

    '''
    Augments given text

    Returns:
    - Tuple, (simple, complex) most simple and most complex forms of text
    '''
    words = nltk.tokenize.word_tokenize(text)
    simple_list = copy.deepcopy(words)
    complex_list = copy.deepcopy(words)

    # Creates a list of all words with parts of speech
    tags = nltk.pos_tag(words)
    tag_list = list(map(list, tags))

    # Finds the simplest word and most complex word for certain parts of speech
    for i, tag in enumerate(tag_list):
        if speech_part(tag[1]):
            num_syllables = analyze.syllables_in_word(tag[0])
            if num_syllables > 0:
                word = tag[0]
                word_options = []
                syns = wordnet.synsets(word)
                for syn in syns:
                    hyps = syn.hypernyms()
                    if len(hyps) != 0:
                        word_options += hyps[0].lemma_names()
                # [[[first, second], syllables]]
                word_options = [[[opt]] for opt in word_options]

                # handles synonyms that are multiple words seperated by '_'
                for option in word_options:
                    total_syl = []
                    if option[0][0].find("_"):
                        option[0] = option[0][0].split("_")
                    for opt in option[0]:
                        total_syl.append(analyze.syllables_in_word(opt))
                    option.append(max(total_syl))

                word_options = sorted(word_options, key=lambda x:x[1])

                # replaces the original word if new is more simple/complex
                if len(word_options) > 0:
                    if word_options[0][1] < num_syllables:
                        for opt in word_options[0][0]:
                            simple_list[i] = opt + " "
                        simple_list[i] = simple_list[i][:-1]
                    else:
                        simple_list[i] = words[i]
                    if word_options[len(word_options) - 1][1] > num_syllables:
                        for opt in word_options[len(word_options) - 1][0]:
                            complex_list[i] = opt + " "
                        complex_list[i] = complex_list[i][:-1]
                    else:
                        simple_list[i] = words[i]

    simple_sent = sentence_join(simple_list)
    complex_sent = sentence_join(complex_list)
    return (simple_sent, complex_sent)


if __name__ == "__main__":
    text_examples = []
    text_examples.append("In the beginning God created the heavens and the earth. Now the earth was formless and empty, darkness was over the surface of the deep, and the Spirit of God was hovering over the waters. And God said, “Let there be light,” and there was light. God saw that the light was good, and he separated the light from the darkness. God called the light “day,” and the darkness he called “night.” And there was evening, and there was morning — the first day.")
    text_examples.append("It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way — in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.")
    text_examples.append("I kissed a girl and I liked it. The taste of her cherry chap stick. I kissed a girl just to try it. I hope my boyfriend dont mind it. It felt so wrong. It felt so right. Dont mean Im in love tonight. I kissed a girl and I liked it. I liked it.")
    text_examples.append("The industrial revolution and its consequences have been a disaster for the human race. They have greatly increased the life-expectancy of those of us who live in \"advanced\" countries, but they have destabilized society, have made life unfulfilling, have subjected human beings to indignities, have led to widespread psychological suffering (in the Third World to physical suffering as well) and have inflicted severe damage on the natural world. The continued development of technology will worsen the situation. It will certainly subject human beings to greater indignities and inflict greater damage on the natural world, it will probably lead to greater social disruption and psychological suffering, and it may lead to increased physical suffering even in \"advanced\" countries.")
    text_examples.append("Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But in a larger sense, we cannot dedicate – we cannot consecrate – we cannot hallow – this ground.")

    for text in text_examples:
        text_aug = augment(text)
        print(text)
        print("Score: " + str(analyze.readability_score(text)))
        print("-----------")
        print(text_aug[0])
        print("Score: " + str(analyze.readability_score(text_aug[0])))
        print("-----------")
        print(text_aug[1])
        print("Score: " + str(analyze.readability_score(text_aug[1])))
        print("-----------")
    exit()
