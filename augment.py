#!/usr/bin/python3
'''
Augments the readability of some texts.

CIS 365 - Artificial Intelligence - Project 2
Frank Wanye
Kellin McAvoy
Andrew Prins
Troy Madsen
'''

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

if __name__ == "__main__":
    text = "The history of all hitherto existing societies is the history of class struggles. Freeman and slave, patrician and plebeian, lord and serf, guild-master and journeyman, in a word, oppressor and oppressed, stood in constant opposition to one another, carried on an uninterrupted, now hidden, now open fight, a fight that each time ended, either in a revolutionary re-constitution of society at large, or in the common ruin of the contending classes. In the earlier epochs of history, we find almost everywhere a complicated arrangement of society into various orders, a manifold gradation of social rank. In ancient Rome we have patricians, knights, plebeians, slaves; in the Middle Ages, feudal lords, vassals, guild-masters, journeymen, apprentices, serfs; in almost all of these classes, again, subordinate gradations."
    words = nltk.tokenize.word_tokenize(text)
    words = list(filter(lambda word : word not in WORD_EXCEPTIONS, words))

    # Creates a list of all words with parts of speech
    tags = nltk.pos_tag(words)

    tag_list = map(list, tags)
    editable = []
    not_editable = []
    text_split = text.split()
    print(text_split)

    # Picks out words with the part of speech that meet certain criteria
    for i, tag in enumerate(tag_list):
        print(i, tag[0])
        text_split.append(tag[0])
        if speech_part(tag[1]):
            syl = analyze.syllables_in_word(tag[0])
            tag.append(syl)
            editable.append(tag)
            not_editable.append(-1)
        else:
            not_editable.append(i)
    print(editable)
    print(not_editable)
    print(text_split)
    print("-----------")

    text_long = text_short = ""
    edits = []

    # Finds all possible synonyms for each editable word
    for word in editable:
        print("Original: " + word[0])
        choices = []
        syns = wordnet.synsets(word[0])
        for syn in syns:
            hyps = syn.hypernyms()
            if len(hyps) != 0:
                choices += hyps[0].lemma_names()
        choice_syl = []
        for choice in choices:
            choice_syl.append([choice, analyze.syllables_in_word(choice)])
        shortest = word[0]
        s_syl = l_syl = word[2]
        longest = word[0]
        for choice in choice_syl:
            if choice[1] > l_syl:
                l_syl = choice[1]
                longest = choice[0]
            elif choice[1] < s_syl:
                s_syl = choice[1]
                shortest = choice[0]
        edits.append([word[0], shortest, longest])
        print("Shortest: " + shortest)
        print("Longest: " + longest)
        print("#####")
    count = 0
    print(edits)
    print("#####")

    #Recombines sentences
    for word in not_editable:
        if word == -1:
            text_short += edits[count][1]
            text_long += edits[count][2]
            count += 1
        else:
            text_short += text_split[word]
            text_long += text_split[word]
        text_short += " "
        text_long += " "
    print(text)
    print("SCORE: " + str(analyze.readability_score(text)))
    print("#####")
    print(text_long)
    print("SCORE: " + str(analyze.readability_score(text_long)))
    print("#####")
    print(text_short)
    print("SCORE: " + str(analyze.readability_score(text_short)))
