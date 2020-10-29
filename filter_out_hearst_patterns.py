'''
First, run conda activate hugface
'''

import pickle
import pdb
import stanza


############### LOAD UKWAC OR FLICKR SENTENCES

corpus = "ukwac"
#corpus = "flickr"

if corpus == "ukwac":
    sentences_filename = "unfiltered_ukwac_scalar_sentences_for_lm.pkl"
elif corpus == "flickr":
    sentences_filename = "unfiltered_flickr_scalar_sentences_for_lm.pkl"


out_filename = "patternfiltered_" + corpus + "_scalar_sentences_for_lm.pkl"
dict_for_lm = pickle.load(open(sentences_filename, "rb"))


nlp = stanza.Pipeline('en')

sentences_patterns = dict()
sentences_patterns["andorother"] = []
sentences_patterns["Ysuchas"] = []
sentences_patterns["suchYas"] = []
sentences_patterns["including"] = []
sentences_patterns["especially"] = []
sentences_patterns["like"] = []


new_dict = dict()

j = 0
for sc in dict_for_lm:
    j+=1
    #if j % 5 == 0:
    print(j)
    for w in dict_for_lm[sc]: #word_sentence_dict:
        for instance in dict_for_lm[sc][w]: #word_sentence_dict[w]:
            l = instance["sentence_words"]
            i = instance["position"]
            ### To save time and computation, before parsing the sentence, check if contains the words that can be part of hearst patterns:
            if not ("other" in l[instance["position"]:] or "including" in l[instance["position"]:] or "especially" in l[instance["position"]:] or "such" in l or "like"[instance["position"]:] in l):
                if sc not in new_dict: # add it in the newdict
                    new_dict[sc] = dict()
                if w not in new_dict[sc]:
                    new_dict[sc][w] = []
                new_dict[sc][w].append(instance)
                continue # and keep going
            


            sentence_string = " ".join(l)
            doc = nlp(sentence_string)
            found = False
            # FIRST, locate your adjective and the noun it modifies.
            startcharadj = sum([len(w) for w in l[:i]]) + i # i is the number of blanks that I will have inserted
            endcharadj = startcharadj + len(l[i])
            my_adjective = '' # index of my adjective in the dependency parsed sentence
            for sentence in doc.sentences:
                for word in sentence.words:
                    if word.text == l[i]:
                        miscsdict = dict()
                        misc = word.misc.split("|")
                        for m in misc:
                            m = m.split("=")
                            miscsdict[m[0]] = m[1]
                        if int(miscsdict["start_char"])  == startcharadj and int(miscsdict["end_char"]) == endcharadj:
                            my_adjective = word.id
            # Locating the noun it modifies:
            my_noun = ''
            if not my_adjective:
                if w not in ["silent", "ill"]:
                    pdb.set_trace() ################# silent -> sile nt 
                continue
            for head, rel, modifier in doc.sentences[0].dependencies:
                if modifier.id == my_adjective and rel == "amod":
                    my_noun = head.id
            #pdb.set_trace()

            conditions = dict()
            ###### AND OTHER, OR OTHER:
            ### my_noun is in "conj" with something, and/or is doing cc to your noun, and the word "other" doing "amod" to your noun.
            ###### INCLUDING:
            ### some noun is doing "nmod" to my noun, and that noun is modified by "including" with the relation "case", and 
            ###### ESPECIALLY:
            ### some noun is doing "appos" to my noun, and  that noun is modified by "especially" with the relation "appos"
            ###### Y SUCH AS X
            ### some noun is doing "nmod" to my noun, and that noun is modified by such with the relation "case", and #### "as" modifies "such" with the relation "fixed"
            ###### SUCH Y AS X
            ### "such" is modifying businesses with "amod" # and there is an "as" closely after my noun...
            ###### LIKE
            ### some noun does "nmod" to my noun and "like" does "case" to this other noun.
            found_conditions = set()
            conditions["andorother"] = set(["conj", "andor", "other"])
            conditions["including"] = set(["nmod","includingcase"])
            conditions["especially"] = set(["appos","especiallyadvmod"])
            conditions["Ysuchas"] = set(["nmod", "suchcase"]) # big animals such as elephants...
            conditions["suchYas"] = set(["suchamod_mynoun", "closeas"]) # I need an "as" after my noun, but I can't trust its label...
            conditions["like"] = set(["likecase"])

            for head, rel, modifier in doc.sentences[0].dependencies:
                if modifier.id == my_noun and rel == "conj": ###### If my_noun is in conj with something (andother)
                    ### see if there is a word and/or doing cc to your noun, and the word "other" doing "amod" to your noun.
                    found_conditions.add("conj")
                if head.id == my_noun and modifier.text in ["and","or"] and rel == "cc": ### see if there is a word "and/or" doing cc to your noun, (andother)
                    found_conditions.add("andor")
                if head.id == my_noun and modifier.text == "other" and rel == "amod": ## see if there is the word "other" doing "amod" to your noun. (andother)
                    found_conditions.add("other")
                if head.id == my_noun and rel == "nmod":
                    other_noun = modifier.id
                    found_conditions.add("nmod")
                    for head, rel, modifier in doc.sentences[0].dependencies:
                        if head.id == other_noun and rel == "case" and modifier.text == "including":
                            found_conditions.add("includingcase")
                        if head.id == other_noun and rel == "case" and modifier.text == "such":
                            found_conditions.add("fixedcase")
                        if head.id == other_noun and rel == "case" and modifier.text == "like":
                            found_conditions.add("likecase")

                if head.id == my_noun and rel == "appos":
                    other_noun = modifier.id
                    found_conditions.add("appos")
                    for head, rel, modifier in doc.sentences[0].dependencies:
                        if head.id == other_noun and rel == "advmod" and modifier.text == "especially":
                            found_conditions.add("especiallyadvmod")
                if head.id == my_noun and rel == "amod" and modifier.text == "such":
                    found_conditions.add("suchamod_mynoun")

            if "suchamod_mynoun" in found_conditions: # look for a close as
                if "as" in l[i:i+4]:
                    found_conditions.add("closeas")

            # now see if you found any pattern
            found = False
            for pattern in conditions:
                if conditions[pattern].issubset(found_conditions):
                    sentences_patterns[pattern].append((w, l, i))
                    found = True
            if not found:
                if sc not in new_dict:
                    new_dict[sc] = dict()
                if w not in new_dict[sc]:
                    new_dict[sc][w] = []
                new_dict[sc][w].append(instance)

#    break



for k in sentences_patterns:
    print(k, len(sentences_patterns[k]))


pickle.dump(new_dict, open(out_filename, "wb"))


############ OLD CODE BASED ON POSITIONS:

'''
for sc in dict_for_lm:
    for w in dict_for_lm[sc]: #word_sentence_dict:
        for instance in dict_for_lm[sc][w]: #word_sentence_dict[w]:
            l = instance["sentence_words"]
            tokens = instance["sentence_tokens"]
            i = instance["position"]
            found = False
            #pdb.set_trace()
            if l[i-2:i] == ["and", "other"]:
                sentences_patterns["andother"].append((w, l, tokens, i))
                found = True
            if l[i-2:i] == ["or", "other"]:
                sentences_patterns["orother"].append((w, l, tokens, i))
                found = True
            if len(l) > i + 4 and l[i+1:i+3] == ["such", "as"]:
                sentences_patterns["Ysuchas"].append((w, l, tokens, i))
                found = True
            if len(l) > i + 3 and l[i-1] == "such" and l[i+2] == ["as"]: # SUCH big(i) companies AS
                sentences_patterns["suchYas"].append((w, l, tokens, i))
                found = True
            if len(l) > i + 4 and ((l[i+2] == "including" and tokens[i+1][1].startswith("NN") and tokens[i+3][1] != "IN" and not tokens[i+3][1].startswith("V")) or (l[i+3] == "including")  and tokens[i+1][1].startswith("NN") and tokens[i+4][1] != "IN" and not tokens[i+4][1].startswith("V")):
                sentences_patterns["including"].append((w, l, tokens, i))
                found = True
            if len(l) > i + 4 and ((l[i+2] == "especially" and tokens[i+1][1].startswith("NN") and tokens[i+3][1] != "IN" and not tokens[i+3][1].startswith("V")) or (l[i+3] == "especially")  and tokens[i+1][1].startswith("NN") and tokens[i+4][1] != "IN" and not tokens[i+4][1].startswith("V")):
                sentences_patterns["especially"].append((w, l, tokens, i))
                found = True
            if len(l) > i + 3 and ((l[i+2] == ["like"] and tokens[i+2][1] == "IN") or (l[i+3] == ["like"] and tokens[i+3][1] == "IN")):
                sentences_patterns["like"].append((w, l, tokens, i))
                found = True
            if not found:
                if sc not in new_dict:
                    new_dict[sc] = dict()
                if w not in new_dict[sc]:
                    new_dict[sc][w] = []
                new_dict[sc][w].append(instance)
'''                

