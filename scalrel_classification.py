'''
CHANGE THE WAY OF READING THE DATASET...
I NEED TO EITHER SHARE relational/relational_ctxtembeds_" + args.bpe_str + "_includingrefadjs.pkl","rb"))
OR TO INCLUDE THE SCRIPT TO EXTRACT IT
AND I NEED TO SHARE THE RELATIONAL SENTENCES TOO!!

#### LOADING DATA AND VECTORS
    data_and_vectors = pickle.load(open("relational/relational_ctxtembeds_" + args.bpe_str + "_includingrefadjs.pkl","rb"))
    static_vectors = Magnitude("/vol/work/gari/Resources/FastText/cc.en.300.magnitude")

    #### CALCULATING DIFFVECS
    reference_dataset = "crowd"
    scalar_sentence_data = pickle.load(open("multilingual_experiments/representations/ctxtembeds_en_" + args.bpe_str + "_ukwac-random.pkl",
        "rb"))


        CHANGE NAME OF THE FUNCION CALCUATE_EXTREME_VECTOR. NO LONGER EXTREME. JUST PROTO....
'''


import pickle
from predict import calculate_diff_vector_singlescales, calculate_static_diffvector_singlescales, load_frequency, load_rankings, extract_gold_mildest_extreme
import numpy as np
import random
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse
from pymagnitude import *
import pdb
from nltk.corpus import wordnet as wn
import math

random.seed(7)

def assign_class(data_and_vectors, adj):
    clas = data_and_vectors[adj]["class"]
    if clas == "SCALAR":
        return 1
    elif clas == "RELATIONAL":
        return 0


def calculate_protoadj_vector(scalar_sentence_data, proto_word="good"):
    if extreme_word == "good":
        scale = ("good", "better","remarkable","exceptional", "perfect")
    extr_vectors_by_layer = dict()
    for layer in range(1, 13):
        extr_vectors_by_layer[layer] = []

    for instance in scalar_sentence_data[scale][:10]:
        for layer in range(1, 13):
            proto_rep = instance['representations'][proto_word][layer].numpy()
            extr_vectors_by_layer[layer].append(proto_rep)

    final_extr_vector_by_layer = dict()
    for layer in extr_vectors_by_layer:
        final_extr_vector_by_layer[layer] = np.average(extr_vectors_by_layer[layer], axis=0)

    return final_extr_vector_by_layer


def prepare_features(data_and_vectors, adj_split, scalar_sentence_data, static_vectors, method="diffvec1neutralscale_X10"):
    if method == "freq":
        freq_counts = load_frequency("en")
    elif method == 'diffvec1scale_X10':
        diffvecs = calculate_diff_vector_singlescales(scalar_sentence_data, method=method, reference_dataset="crowd", language="en")
    elif method == 'staticdiffvec1scale':
        diffvecs = calculate_static_diffvector_singlescales(static_vectors, method=method, reference_dataset="crowd",language="en")
    elif "proto-adj" in method:
        diffvecs = calculate_protoadj_vector(scalar_sentence_data, proto_word=method.split("_")[1])

    features_by_layer_and_subset = dict()
    if not "static" in method and method not in ["freq", "sense"]:
        for layer in range(1, 13):
            features_by_layer_and_subset[layer] = dict()
            if "diffvec" in method or "proto-adj_" in method:
                dvec = diffvecs[layer]
            for subset in adj_split:
                X, y = [], []
                for adj in adj_split[subset]:
                    adj_embeddings = []
                    for s in data_and_vectors[adj]["sentences"][:10]:
                        adj_embeddings.append(np.array(s["representations"][layer]))
                    adj_embedding = np.average(adj_embeddings, axis=0)

                    if method == "adj-rep":
                        feature = adj_embedding
                    elif "diffvec" in method:
                        feature = np.abs(1 - cosine(adj_embedding, dvec))
                    elif "proto-adj" in method:
                        feature = 1-cosine(adj_embedding, dvec)
                    X.append(feature)
                    y.append(assign_class(data_and_vectors, adj))

                if "diffvec" in method or "proto-adj" in method:
                    X = np.array(X).reshape(-1,1)
                features_by_layer_and_subset[layer][subset] = (X, y)

    elif "static" in method or method in ["freq","sense"]:
        features_by_layer_and_subset[0] = dict()
        if "diffvec" in method:
            dvec = diffvecs
        for subset in adj_split:
            X, y = [], []
            for adj in adj_split[subset]:
                y.append(assign_class(data_and_vectors, adj))
                if "static" in method:
                    adj_embedding = static_vectors.query(adj)
                if method == "static-adj-rep":
                    feature = adj_embedding
                elif "diffvec" in method:
                    feature = np.abs(1-cosine(adj_embedding, dvec))
                elif method == "freq":
                    feature = math.log(freq_counts[adj]) if freq_counts[adj] > 0 else -1
                elif method == "sense":
                    feature = len(wn.synsets(adj))
                X.append(feature)
            if "diffvec" in method or method in ["freq", "sense"]:
                X = np.array(X).reshape(-1,1)
            features_by_layer_and_subset[0][subset] = (X, y)

    return features_by_layer_and_subset


def load_adj_split(fn="scal-rel/scal-rel_dataset.csv"):
    adj_split = dict()
    for subset in ["train","dev","test"]:
        adj_split[subset] = []
    with open(fn) as f:
        for l in f:
            adj, cl, subset = l.strip().split("\t")
            adj_split[subset].append(adj)
    return adj_split



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_last_bpe", action="store_true", help="whether we exclude the last piece when a word is split into multiple wordpieces."
                             "Otherwise, we use the representations of all pieces.")
    parser.add_argument("--path_to_static", default="", type=str, help="The path to the directory containing magnitude fasttext embeddings")
    args = parser.parse_args()

    if not args.exclude_last_bpe:
        args.bpe_str = "all-bpes"
    else:
        args.bpe_str = "exclude-last-bpes"

    #### LOADING DATA AND VECTORS
    data_and_vectors = pickle.load(open("relational_ctxtembeds_" + args.exclude_last_bpe + ".pkl","rb"))
    static_vectors = Magnitude(args.path_to_static + "cc.en.300.magnitude")

    #### Data to create diffvec
    scalar_sentence_data = pickle.load(open("multilingual_experiments/representations/ctxtembeds_en_" + args.exclude_last_bpe + "_ukwac-random.pkl", "rb"))
    diffvecs = dict()
    staticdiffvecs = dict()

    methods = ['diffvec1scale_X10', 'staticdiffvec1scale', 'adj-rep', "static-adj-rep", "sense", "freq", 'proto-adj_good']

    adj_split = load_adj_split()

    for method in methods:
        print("*********", method, "***********")
        features = prepare_features(data_and_vectors, adj_split, scalar_sentence_data, static_vectors, method=method)

        res_by_layer = dict()
        pred_by_layer = dict()

        for layer in features:
            res_by_layer[layer] = dict()
            pred_by_layer[layer] = dict()
            logreg = LogisticRegression(solver='lbfgs')
            logreg.fit(features[layer]["train"][0], features[layer]["train"][1])
            dev_predictions = logreg.predict(features[layer]["dev"][0])
            test_predictions = logreg.predict(features[layer]["test"][0])
            dev_acc = accuracy_score(features[layer]["dev"][1], dev_predictions)
            test_acc = accuracy_score(features[layer]["test"][1], test_predictions)
            res_by_layer[layer]['dev'] = dev_acc
            res_by_layer[layer]['test'] = test_acc
            pred_by_layer[layer]['dev'] = dev_predictions
            pred_by_layer[layer]['test'] = test_predictions
        maxdevacc = max([res_by_layer[layer]['dev'] for layer in res_by_layer])
        for layer in res_by_layer:
            if res_by_layer[layer]['dev'] == maxdevacc:
                print("layer:", layer, "dev acc:", res_by_layer[layer]['dev'], "test acc:", res_by_layer[layer]['test'])