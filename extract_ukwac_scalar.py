'''
Extracts sentences from ukwac containing scalar adjectives
'''

import gzip
from read_scalar_datasets import read_scales
import pickle
from string import punctuation
import argparse


def check_interesting_sentence(sentence, relevant_words):
    interesting = False
    present_words = []
    position = 0
    for wordposlemma in sentence:
        if len(wordposlemma) != 3:
            continue
        word, pos, lemma = wordposlemma
        if lemma in relevant_words and posmap(pos) == relevant_words_dict[lemma]:
            present_words.append((lemma, position))
            interesting = True
        position += 1
    if interesting:
        return present_words


def accepted_pos(pos):
    if "JJ" in pos or "RB" in pos or "DT" in pos or "VV" in pos:
        return True
    return False


class IterCorpus():
    '''Iterator class that iterates over sentences, which
    are separated by a space.
    '''

    def __init__(self, filename):
        self.file = gzip.open(filename).read().decode("latin-1").strip().split("\n")
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file[self.index].strip()  # self.file.readline()

        sentence = []
        while line != "</s>" and line != "":
            if line != "<s>" and not line.startswith("<text id=") and not line.startswith("</text"):
                sentence.append(tuple(line.split("\t")))
            self.index += 1
            if self.index == len(self.file):
                raise StopIteration()
            else:
                line = self.file[self.index]  # self.file.readline()
        if line == "</s>":
            self.index += 1
            return tuple(sentence)

        elif line == "":
            raise StopIteration()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", type=str, help="Folder containing the scalar adjective datasets.")
    parser.add_argument("--corpus_dir", default="ukWaC/", type=str,help="Folder containing the ukwac corpus")
    args = parser.parse_args()


    #### Read scales
    rankings = dict()
    datanames = ["demelo", "crowd", "wilkinson"]
    for dataname in datanames:
        r = read_scales(args.data_dir + dataname + "/gold_rankings/")
        rankings[dataname] = r

    my_words = set()
    for dataname in rankings:
        for scale in rankings[dataname]:
            for word in rankings[dataname][scale]:
                words = word.split(" || ")
                for w in words:
                    my_words.add(w)


    word_sentence_dict = dict()
    for word in my_words:
        word_sentence_dict[word] = set()



    chars_to_remove = ["\x80", "\x91", "\x93", "\x94", "\x96", "\x95","\x98","\x99","\x9c","\x92","\x97","\x85", r"\xao"]


    last_file_number = 25 # number of the last ukwac file to read
    prefix = "UKWAC-"
    suffix = ".xml.gz"

    sentences = set()
    j = 0
    for i in range(1, last_file_number + 1):
        if not my_words: # If I already found enough sentences for all of my words, stop the loop
            break
        filename = args.corpus_dir + prefix + str(i) + suffix
        print("Doing file", filename)
        print("Pending words:", len(my_words))
        for sentence_tokens in IterCorpus(filename):
            if not my_words:
                break
            if len(sentence_tokens) > 100 or len(sentence_tokens) < 5: # not too long, not too short
                continue
            discard = False
            for token in sentence_tokens: #avoid sentences containing urls
                if "http:" in token[0] or "www" in token[0]:
                    discard = True
                    break
                for p in punctuation:
                    if p in token[0] and token[1] in ["NN","VV","RB","JJ"]:
                        discard = True
                        break

                for wc in chars_to_remove:
                    if wc in token[0]:
                        discard = True
                        break
                if len(token) != 3 or token[0] == "" or token[1] == 'SYM': # tokens that have more than token-pos-lemma.... # SYMS are problematic (copyright symbol,  |, floating dot, asterisc...)
                    discard = True
                    break
            if discard:
                continue
            sentence_words = [x[0] for x in sentence_tokens]
            for i, token in enumerate(sentence_tokens):
                word, pos, lemma = token
                if word in my_words and accepted_pos(pos):
                    word_sentence_dict[word].add((tuple(sentence_words), tuple(sentence_tokens), i))
                    if len(word_sentence_dict[word]) >= 1000: # don't collect more than 1000 per adj
                        my_words.remove(word)



    # Save them all, later they will be filtered & selected (randomly or with c2v)
    dict_for_lm = dict()

    for dataname in rankings:
        for scale in rankings[dataname]:
            words_in_scale = []
            for ws in rankings[scale]:
                words_in_scale.extend(ws.split(" || "))
            words_in_scale = tuple(words_in_scale)
            dict_for_lm[words_in_scale] = dict()

            for word in word_sentence_dict:
                if word in words_in_scale:
                    dict_for_lm[words_in_scale][word] = []
                    for sentence_words,sentence_tokens, position in word_sentence_dict[word]:
                        instance = dict()
                        instance['sentence_words'] = sentence_words
                        instance['sentence_tokens'] = sentence_tokens
                        instance['position'] = int(position)
                        dict_for_lm[words_in_scale][word].append(instance)

    pickle.dump(dict_for_lm, open("all_unfiltered_ukwac_scalar_sentences.pkl", "wb"))


