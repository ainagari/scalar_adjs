from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
import torch
import numpy as np
import pickle
import argparse
from extract_representations import aggregate_reps, special_tokenization

def check_correct_token_mapping(bert_tokenized_sentence, positions, word):
    berttoken = ''
    for p in positions:
        berttoken += bert_tokenized_sentence[p].strip("##")
    if berttoken.lower() == word.lower():
        return True
    else:
        return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_last_bpe", action="store_true", type=str,
                        help="whether we exclude the last piece when a word is split into multiple wordpieces."
                             "Otherwise, we use the representations of all pieces.")
    args = parser.parse_args()

    if not args.exclude_last_bpe:
        bpe_str = "all-bpes"
    else:
        bpe_str = "exclude-last-bpes"

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    filename = "scal-rel/relational_sentences.pkl"
    out_fn = "relational_ctxtembeds_" + bpe_str + ".pkl"

    data = pickle.load(open(filename, "rb"))

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1  # do not change this


    infos = []
    for adj in data:
        for instance in data[adj]['sentences'][:10]:
            sentence_words = instance['sentence_words']
            if '' in sentence_words:
                print(sentence_words)
            bert_tokenized_sentence, mapp = special_tokenization(" ".join(sentence_words), tokenizer, model_name)
            bert_position = mapp[instance['position']]  # this is a list of positions
            if not check_correct_token_mapping(bert_tokenized_sentence, bert_position, adj):
                sys.out("Tokenization mismatch!")
            cinstance = dict()
            cinstance['adj'] = adj
            cinstance['class'] = data[adj]['class']
            cinstance['sentence_words'] = sentence_words
            cinstance["bert_tokenized_sentence"] = bert_tokenized_sentence
            cinstance["bert_position"] = bert_position
            infos.append(cinstance)

    #### EXTRACTING REPRESENTATIONS
    reps = extract_representations(infos, tokenizer, model_name)

    for rep, instance in zip(reps, infos):
        adj = instance["adj"]
        clas = instance["class"]
        for ins2 in data[adj]['sentences']:
            if ins2["sentence_words"] == instance["sentence_words"]:
                if "representations" not in ins2:
                    ins2["representations"] = dict()
                for l in rep:
                    ins2['representations'][l] = aggregate_reps(rep[l], hidden_size=model.config.hidden_size)

    pickle.dump(data, open(out_fn, "wb"))
