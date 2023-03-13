
from transformers import BertTokenizer, BertConfig, BertModel, AutoTokenizer, AutoModel, FlaubertTokenizer, FlaubertModel, AutoConfig, FlaubertConfig
import torch
import numpy as np
import pickle
import sys
from copy import deepcopy
import argparse


def aggregate_reps(reps_list, hidden_size):
    '''This function averages representations of a word that has been split into wordpieces.'''
    reps = torch.zeros([len(reps_list), hidden_size])
    for i, wrep in enumerate(reps_list):
        w, rep = wrep
        reps[i] = rep

    if len(reps) > 1:
        reps = torch.mean(reps, axis=0)
    reps = reps.view(hidden_size)

    return reps.cpu()


def special_tokenization(sentence, tokenizer, model_name):
    map_ori_to_bert = []
    if "flaubert" in model_name:
        tok_sent = ['<s>']
    else:
        tok_sent = ['[CLS]']

    for orig_token in sentence.split():
        current_tokens_bert_idx = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token) # tokenize
        tok_sent.extend(bert_token) # add to my new tokens
        if len(bert_token) > 1: # if the new token has been 'wordpieced'
            extra = len(bert_token) - 1
            for i in range(extra):
                current_tokens_bert_idx.append(current_tokens_bert_idx[-1]+1) # list of new positions of the target word in the new tokenization
        map_ori_to_bert.append(tuple(current_tokens_bert_idx))

    if "flaubert" in model_name:
        tok_sent.append('</s>')
    else:
        tok_sent.append('[SEP]')

    return tok_sent, map_ori_to_bert




def extract_representations(infos, tokenizer, model_name):
    reps = []
    if model_name in ["bert-base-uncased", "bert-base-cased", "bert-base-multilingual-uncased", "bert-base-multilingual-cased"]:
        config_class, model_class = BertConfig, BertModel        
    elif "flaubert" in model_name:
        config_class, model_class = FlaubertConfig, FlaubertModel
    elif "greek" in model_name or "spanish" in model_name:
        config_class, model_class = AutoConfig, AutoModel

    config = config_class.from_pretrained(model_name, output_hidden_states=True)
    model = model_class.from_pretrained(model_name, config=config)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for info in infos:
            tok_sent = info['bert_tokenized_sentence']            
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tok_sent)]).to(device)            
            outputs = model(input_ids)            
            if "flaubert" in model_name:
                hidden_states = outputs[1]
            else:
                hidden_states = outputs[2]
            if not args.exclude_last_bpe:
                bpositions = info["bert_position"]
            else:
                if len(info["bert_position"]) == 1:
                    bpositions = info["bert_position"]
                if len(info["bert_position"]) > 1:
                    bpositions = info["bert_position"][:-1]                    
            
            reps_for_this_instance = dict()                
            for i, w in enumerate(info["bert_tokenized_sentence"]):
                if i in bpositions: 
                    for l in range(len(hidden_states)): # all layers
                        if l not in reps_for_this_instance:
                            reps_for_this_instance[l] = []
                        reps_for_this_instance[l].append((w, hidden_states[l][0][i].cpu()))                        
            reps.append(reps_for_this_instance)            

    return reps, model




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/', type=str, help="directory where sentences are stored and where representations will be saved")
    parser.add_argument("--language", default="en", type=str, help="Language of the sentences: en, es, el, fr.")
    parser.add_argument("--multilingual_uncased", action="store_true", help="Whether we use multilingual BERT uncased. If no multilingual model is chosen, the monolingual BERT of the chosen language will be used.")
    parser.add_argument("--multilingual_cased", action="store_true", help="Whether we use multilingual BERT cased. If no multilingual model is chosen, the monolingual BERT of the chosen language will be used.")
    parser.add_argument("--sentences", default='ukwac-random', type=str, help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'ukwac-random'")
    parser.add_argument("--exclude_last_bpe", action="store_true", help="whether we exclude the last bpe of words when words are split into multiple wordpieces")

    args = parser.parse_args()

    if args.multilingual_uncased and args.multilingual_cased:
        sys.out("incompatible options")
        

    if args.multilingual_uncased:
        language_str = "multi-" +  args.language 
    elif args.multilingual_cased:
        language_str = "multicased-" + args.language
    else:
        language_str = args.language


    if not args.exclude_last_bpe:
        bpe_str = "all-bpes"
    else:
        bpe_str = "exclude-last-bpes"


    if args.language == "en" and not args.multilingual_uncased and not args.multilingual_cased:
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        in_fn = args.data_dir + args.sentences + "_selected_scalar_sentences.pkl"
        

    elif args.language in ["es","el", "fr"]:
       
        args.sentences = "oscar"
        in_fn = args.data_dir + "sentences_" + args.language + ".pkl"
        

        if not args.multilingual_uncased and not args.multilingual_cased:
            if args.language == "el":
                model_name = "nlpaueb/bert-base-greek-uncased-v1"
                tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
            elif args.language == "es":
                model_name = "dccuchile/bert-base-spanish-wwm-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
            elif args.language == "fr":
                model_name = "flaubert/flaubert_base_uncased"
                tokenizer = FlaubertTokenizer.from_pretrained(model_name, do_lower_case=True)      

    
    out_fn = args.data_dir + "scalar_embeddings_" + language_str + "_" + bpe_str + "_" + args.sentences + ".pkl"        
    
        
    if args.multilingual_uncased:
        model_name = "bert-base-multilingual-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    elif args.multilingual_cased:
        model_name = "bert-base-multilingual-cased"
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)



    data = pickle.load(open(in_fn, "rb"))

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1 


    infos = []
    for scale in data:
        for instance in data[scale]:            
            if '' in instance['sentence_words']:
                print(instance['sentence_words'])
            for scaleword in scale:
                cinstance = deepcopy(instance) 
                sentence_words = list(cinstance["sentence_words"][:])

                if args.language == "en":
                    # Replace a by an and viceversa if necessary
                    if sentence_words[cinstance["position"]-1] == "a" and scaleword[0] in 'aeiou':
                        sentence_words[cinstance["position"]-1] = "an"
                    elif sentence_words[cinstance["position"]-1] == "an" and scaleword[0] not in 'aeiou':
                        sentence_words[cinstance["position"]-1] = "a"
                    
                    # Replace original adjective by current adjective
                    sentence_words[cinstance["position"]] = scaleword                                
                    cinstance["position"] = [cinstance["position"]]

                elif args.language in ["el","es", "fr"]:
                    # take care of multiple word adjectives ("bien parecido")
                    original_positions = instance['position']                    
                    original_word = [instance['sentence_words'][p] for p in original_positions]                
                    sentence_words = sentence_words[:original_positions[0]] + scaleword.split() + sentence_words[original_positions[0]+len(original_word):]                    
                    if len(scaleword.split()) == 1: # if it is a phrase
                        if len(original_word) > 1:                            
                            cinstance['position'] = [original_positions[0]]                            
                    elif len(scaleword.split()) > 1: # if the substitute is a phrase                                                
                        cinstance['position'] = [p for p in range(original_positions[0], original_positions[0]+len(scaleword.split()))]                                                                

                
                bert_tokenized_sentence, mapp  = special_tokenization(" ".join(sentence_words), tokenizer, model_name)                  
                current_positions = cinstance['position']
                
                if len(current_positions) == 1:
                    bert_position = mapp[cinstance['position'][0]] # this is a list of positions (it might have been split into wordpieces)
                elif len(current_positions) > 1:
                    bert_position = []
                    for p in current_positions:
                        bert_position.extend(mapp[p])

                cinstance["bert_tokenized_sentence"] = bert_tokenized_sentence
                cinstance["bert_position"] = bert_position
                cinstance["scale"] = scale
                cinstance["lemma"] = scaleword
                infos.append(cinstance)

    #### EXTRACTING REPRESENTATIONS
    reps, model = extract_representations(infos, tokenizer, model_name)

    for rep, instance in zip(reps, infos):
        scale = instance["scale"]
        lemma = instance["lemma"]
        for ins2 in data[scale]:
            if ins2["sentence_words"] == instance["sentence_words"]:
                if "representations" not in ins2:
                    ins2["representations"] = dict()
                if lemma not in ins2["representations"]:
           	       	ins2["representations"][lemma] = dict()
                for l in rep:                    
                    ins2['representations'][lemma][l] = aggregate_reps(rep[l], model.config.hidden_size)


    pickle.dump(data, open(out_fn, "wb"))
