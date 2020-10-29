
from transformers import BertTokenizer, BertConfig, BertModel
import torch
import pickle
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


def special_tokenization(sentence, tokenizer):
    map_ori_to_bert = []

    tok_sent = ['[CLS]']

    for orig_token in sentence.split():
        current_tokens_bert_idx = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token)  # tokenize
        tok_sent.extend(bert_token)  # add to my new tokens
        if len(bert_token) > 1:  # if the new token has been 'wordpieced'
            extra = len(bert_token) - 1
            for i in range(extra):
                current_tokens_bert_idx.append(
                    current_tokens_bert_idx[-1] + 1)  # list of new positions of the target word in the new tokenization
        map_ori_to_bert.append(tuple(current_tokens_bert_idx))

    tok_sent.append('[SEP]')

    return tok_sent, map_ori_to_bert


def extract_representations(infos, tokenizer, model):
    reps = []
    model.eval()
    with torch.no_grad():
        for info in infos:
            tok_sent = info['bert_tokenized_sentence']
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tok_sent)]).to(device)  # CHECK THAT THIS WORKS WELL FOR FLAUBERT TOO
            outputs = model(input_ids)
            hidden_states = outputs[2]
            bpositions = info["bert_position"]

            reps_for_this_instance = dict()
            for i, w in enumerate(info["bert_tokenized_sentence"]):
                if i in bpositions:  # info["bert_position"]: #if it's one of the relevan bertpositions
                    for l in range(len(hidden_states)):  # all layers
                        if l not in reps_for_this_instance:
                            reps_for_this_instance[l] = []
                        reps_for_this_instance[l].append((w, hidden_states[l][0][i].cpu()))

            reps.append(reps_for_this_instance)

    return reps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", default='ukwac', type=str,
                        help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'ukwac-random'")
    parser.add_argument("--data_dir", default='data/', type=str,
                        help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'ukwac-random'")
    args = parser.parse_args()

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    in_fn = args.data_dir + args.sentences + "_selected_scalar_sentences.pkl"
    out_fn = args.data_dir + "scalar_embeddings_" + args.sentences + ".pkl"

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

                # Replace a by an and viceversa if necessary
                if sentence_words[cinstance["position"] - 1] == "a" and scaleword[0] in 'aeiou':
                    sentence_words[cinstance["position"] - 1] = "an"
                elif sentence_words[cinstance["position"] - 1] == "an" and scaleword[0] not in 'aeiou':
                    sentence_words[cinstance["position"] - 1] = "a"

                # Replace original adjective by current adjective
                sentence_words[cinstance["position"]] = scaleword

                # BERT tokenization, and extracting position of word in this tokenization
                bert_tokenized_sentence, mapp = special_tokenization(" ".join(sentence_words), tokenizer)
                bert_position = mapp[cinstance['position']]
                cinstance["bert_tokenized_sentence"] = bert_tokenized_sentence
                cinstance["bert_position"] = bert_position
                cinstance["scale"] = scale
                cinstance["lemma"] = scaleword

                infos.append(cinstance)

    #### Extracting BERT representations
    config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    model = BertModel.from_pretrained(model_name, config=config)
    reps = extract_representations(infos, tokenizer, model)


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
                    ins2['representations'][lemma][l] = aggregate_reps(rep[l], hidden_size=model.config.hidden_size)

    pickle.dump(data, open(out_fn, "wb"))
