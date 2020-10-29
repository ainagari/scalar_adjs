
'''
What this script does (should do)
1. Read adjective scales
2. Look for sentences in ukwac that contain the adjectives in these scales
3. Store them by word

I want to use this for a language model.
Out structure:

Ranking (in order): dict
word : list of dicts
sentence: str or list of strs
position: int

after I do the lm thing (before choosing), I want to have:

I will want to get a bunch of sentences
PER RANKING
per word:
list of dicts
sentence: - 
position: - 
probabilities: dict : 
word : prob

I need to have a sentence, the original position, and the lm probability for all words in the ranking.


'''

from read_scalar_datasets import read_scales
import pickle
import pdb
import spacy


nlp = spacy.load("en_core_web_sm")


rankings = dict()
#dirnames = ["anne_data/demelo/gold_rankings/", "anne_data/crowd/gold_rankings/", "anne_data/wilkinson/gold_rankings/"]
datanames = ["demelo", "crowd", "wilkinson"]
for dataname in datanames:
    r = read_scales("anne_data/" + dataname + "/gold_rankings/")
    rankings[dataname] = r
    #print("new additions", len(r))
    #old_len = len(rankings)
    #print("there were", old_len)
    #rankings.update(r)
    #print("now there are", len(rankings), "I thought there would be", old_len + len(r) )


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

def accepted_pos(pos):
    if pos in ["ADJ","ADV", "ADP","VERB","DET"]: # or "DET" in pos or "VERB" in pos:
        return True
    return False

## read sentences until you find the ones you need
flickr_file = "results_20130124.token"
flickr_folder = "/vol/work/gari/Resources/flickr30k/"

num_of_sentences = 0

# loop over files
with open(flickr_folder + flickr_file) as f:
    for l in f:       
        l = l.strip().split("\t")[1]
        sentence_tokens = tuple(l.split())
        if len(sentence_tokens) > 100:
            continue
        # first check if any of my words is present. otherwise is not worth tagging it.
        found = False
        for token in sentence_tokens:
            if token in my_words:
                found = True
                break
        if found:
            doc = nlp(l)
            new_tokenization = []
            for token in doc:
                new_tokenization.append(token.text)
            if "double-decker" in sentence_tokens:
                pdb.set_trace()
            for i, token in enumerate(doc):
                if token.text in my_words and accepted_pos(token.pos_):
                    word_sentence_dict[token.text].add((tuple(new_tokenization), i)) # sentence and position 
                    num_of_sentences +=1
                #elif token.text in my_words and not accepted_pos(token.pos_ ):
                #    pdb.set_trace()


print(num_of_sentences)

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
                for sentence, position in word_sentence_dict[word]:
                    instance = dict()
                    instance['sentence_words'] = sentence #[token.split("_")[0] for token in sentence]
                    instance['position'] = int(position)
                    dict_for_lm[words_in_scale][word].append(instance)

pickle.dump(dict_for_lm, open("unfiltered_flickr_scalar_sentences_for_lm.pkl","rb"))

#2flickr_provant_scalar_sentences_for_lm.pkl", "wb"))








