
from read_scalar_datasets import read_scales
from nltk.corpus import wordnet as wn
import gzip
import pickle
import numpy as np
import sys
from scipy.spatial.distance import cosine
from operator import itemgetter
from collections import defaultdict
from pymagnitude import *
import argparse


def extract_gold_mildest_extreme(ranking):
    ordered_words = []
    for i, w in enumerate(ranking):
        ordered_words.extend(w.split(" || "))
        if i == 0:
            mildest_words = w.split(" || ")
        if i == len(ranking) - 1:  # last element
            extreme_words = w.split(" || ")
    return ordered_words, mildest_words, extreme_words


def calculate_static_diff_vector(vectors, scalar_dataset, dataname, avoid_overlap=True):
    #### dataset used to build diffvec - dataset on which we will make predictions
    relevant_dds = [dataname + "-" + d for d in datanames if d != dataname]

    diffvectors_extreme_by_scale = dict()
    pairs_by_scale = dict()

    for dd in relevant_dds:
        diffvectors_extreme_by_scale[dd] = []
        pairs_by_scale[dd] = []

    for scale in scalar_dataset:
        ordered_words, mildest_words, extreme_words = extract_gold_mildest_extreme(scalar_dataset[scale])
        mildest_word = mildest_words[0]
        extreme_word = extreme_words[0]

        mild_rep = vectors.query(mildest_word)
        extreme_rep = vectors.query(extreme_word)
        diffvec_ex = extreme_rep - mild_rep


        for dd in relevant_dds:
            if avoid_overlap:
                if mildest_word in adjs_by_dataset[dd.split("-")[1]] or extreme_word in adjs_by_dataset[dd.split("-")[1]]:
                    continue
            diffvectors_extreme_by_scale[dd].append(diffvec_ex)


    final_diff_vector = dict()
    for dd in diffvectors_extreme_by_scale:
        final_diff_vector[dd] = np.average(diffvectors_extreme_by_scale[dd], axis=0)

    return final_diff_vector



def assign_reference_scales(reference_dataset, method, language):
    if language == "en":
        if reference_dataset == "wilkinson":
            if method in ["diffvec1scale", "staticdiffvec1scale"]:
                reference_scales = [('good', 'great', 'wonderful', 'awesome')]
            elif method in ["diffvec1scaleneg","staticdiffvec1scaleneg"]:
                reference_scales = [('bad', 'awful', 'terrible', 'horrible')]
            elif method in ["diffvec2scale","staticdiffvec2scale"]:
                reference_scales = [('good', 'great', 'wonderful', 'awesome'), ('bad', 'awful', 'terrible', 'horrible')]
            elif method in ["diffvec5scale","staticdiffvec5scale"]:
                reference_scales = [('good', 'great', 'wonderful', 'awesome'), ('bad', 'awful', 'terrible', 'horrible'),
                                    ('old', 'ancient'), ('pretty', 'beautiful', 'gorgeous'), ('ugly', 'hideous')]
        elif reference_dataset == "crowd":
            if method in ["diffvec1scale", "staticdiffvec1scale"]:
                reference_scales = [("good", "better", "remarkable", "exceptional", "perfect")]
            elif method in ["diffvec1scaleneg", "staticdiffvec1scaleneg"]:
                reference_scales = [("bad","horrific","horrendous")]
            elif method in ["diffvec2scale", "staticdiffvec2scale"]:
                reference_scales = [("good", "better","remarkable","exceptional", "perfect"), ("bad","horrific","horrendous")]
            elif method in ["diffvec5scale","staticdiffvec5scale"]:
                reference_scales = [('good', 'great', 'wonderful', 'awesome'), ('bad', 'awful', 'terrible', 'horrible'),
                                    ('old', 'ancient'), ('pretty', 'beautiful', 'gorgeous'), ('ugly', 'hideous')]
            elif method in ["diffvec1neutralscale","staticdiffvec1neutralscale"]:
                reference_scales = [('relevant', 'crucial')]
            elif method in ["diffvec3scale","staticdiffvec3scale"]:
                reference_scales = [("good", "better","remarkable","exceptional", "perfect"), ("bad","horrific","horrendous"),('relevant', 'crucial')]

    elif language in ["es","el","fr"]:
        goodscale = dict()
        goodscale["es"] = ("bueno", "perfecto")
        goodscale["el"] = ("καλός", "τέλειος")
        goodscale["fr"] = ("bon", "parfait")
        if reference_dataset == "crowd":
            if method in ["diffvec1scale", "staticdiffvec1scale"]:
                reference_scales = [goodscale[language]]
            else:
                sys.out("Not implemented")

    return reference_scales



def calculate_static_diffvector_singlescales(vectors, method="staticdiffvec1scale", reference_dataset="wilkinson", language="en"):
    reference_scales = assign_reference_scales(reference_dataset=reference_dataset, method=method, language=language)

    diffvecs = []

    for scale in reference_scales:
        mild, extreme = scale[0], scale[-1]
        mild_rep = vectors.query(mild)
        extreme_rep = vectors.query(extreme)
        diffvecs.append(extreme_rep - mild_rep)

    final_diff_vector = np.average(diffvecs, axis=0)

    return final_diff_vector


def calculate_diff_vector(data, scalar_dataset, dataname, X=10,  avoid_overlap=True):
    #### dataset used to build diffvec - dataset for which we will make predictions
    relevant_dds = [dataname + "-" + d for d in datanames if d != dataname]

    diff_vectors_by_layer = dict()
    pairs_by_layer = dict()
    for dd in relevant_dds:
        diff_vectors_by_layer[dd] = dict()
        pairs_by_layer[dd] = dict()
        for layer in range(1, 13):
            diff_vectors_by_layer[dd][layer] = []
            pairs_by_layer[dd][layer] = []
    missing = 0

    for scale in scalar_dataset:
        ordered_words, mildest_words, extreme_words = extract_gold_mildest_extreme(scalar_dataset[scale])
        if tuple(ordered_words) not in data:
            print(ordered_words)
            missing += 1
            continue
        mildest_word = mildest_words[0]
        extreme_word = extreme_words[0]
        for dd in relevant_dds:
            if avoid_overlap:
                if mildest_word in adjs_by_dataset[dd.split("-")[1]] or extreme_word in adjs_by_dataset[dd.split("-")[1]]:
                    continue
            diff_vectors_one_scale = dict()
            pairs_one_scale = dict()
            for layer in range(1, 13):
                diff_vectors_one_scale[layer] = []
                pairs_one_scale[layer] = []

            for instance in data[tuple(ordered_words)][:X]:
                for layer in range(1,13):

                    mild_rep = instance['representations'][mildest_word][layer].numpy()
                    extreme_rep = instance['representations'][extreme_word][layer].numpy()
                    diffvec_ex = extreme_rep - mild_rep
                    diff_vectors_one_scale[layer].append(diffvec_ex)

            for layer in range(1, 13):
                av_ex = np.average(diff_vectors_one_scale[layer], axis=0)
                diff_vectors_by_layer[dd][layer].append(av_ex)

    final_diff_vector_by_layer = dict()
    for dd in diff_vectors_by_layer:
        final_diff_vector_by_layer[dd] = dict()
        for layer in range(1,13):
            av_ex = np.average(diff_vectors_by_layer[dd][layer], axis=0)
            final_diff_vector_by_layer[dd][layer] = av_ex

    print('missing scales:', missing)
    return final_diff_vector_by_layer

def calculate_diff_vector_singlescales(data, method="diffvec1scale", X=10, reference_dataset="wilkinson", language="en"):
    reference_scales = assign_reference_scales(reference_dataset=reference_dataset, method=method, language=language)

    diff_vectors_by_layer = dict()
    pairs_by_layer = dict()
    for layer in range(1, 13):
        diff_vectors_by_layer[layer] = []
        pairs_by_layer[layer] = []
    for scale in reference_scales:
        diff_vectors_in_scale = dict()
        pairs_in_scale = dict()
        mild, extreme = scale[0], scale[-1]
        for instance in data[scale][:X]:
            for layer in range(1, 13):
                if layer not in diff_vectors_in_scale:
                    diff_vectors_in_scale[layer] = []
                    pairs_in_scale[layer] = []
                mild_rep = instance['representations'][mild][layer].numpy()
                extreme_rep = instance['representations'][extreme][layer].numpy()
                diffvec_ex = extreme_rep - mild_rep
                diff_vectors_in_scale[layer].append(diffvec_ex)
                pairs_in_scale[layer].append((mild_rep, extreme_rep))

        for layer in range(1, 13):
            av_of_scale = np.average(diff_vectors_in_scale[layer], axis=0)
            diff_vectors_by_layer[layer].append(av_of_scale)

    final_diff_vector_by_layer = dict()
    for layer in range(1, 13):
        av_ex = np.average(diff_vectors_by_layer[layer], axis=0)
        final_diff_vector_by_layer[layer] = av_ex

    return final_diff_vector_by_layer


def load_rankings(data_dir, datanames=["demelo", "crowd", "wilkinson"]):
    rankings = dict()
    adjs_by_dataset = dict()
    for dataname in datanames:
        rankings[dataname] = read_scales(data_dir + dataname + "/gold_rankings/")
    for dataname in rankings:
        adjs_by_dataset[dataname] = set()
        for scale in rankings[dataname]:
            adjs_by_dataset[dataname].update(get_scale_words(rankings[dataname][scale]))

    return rankings, adjs_by_dataset

def get_scale_words(scale):
    words = []
    for w in scale:
        words.extend(w.split(" || "))
    return words

def load_frequency(language, frequency_fn=""):
    if language == "en":
        freq_counts = defaultdict(int)

        with gzip.open(frequency_fn) as f:
            bytecontents = f.read()
        contents = bytecontents.decode("utf-8")
        contents = contents.split("\n")

        for tokencount in contents:
            s = tokencount.strip().split("\t")
            if len(s) == 2:
                token, count = s
                freq_counts[token] = int(count)

    elif language in ["es","el","fr"]:
        frequency_fn = "multilingual_experiments/freq_counts_"+language+".pkl"
        freq_counts = pickle.load(open(frequency_fn, "rb"))

    return freq_counts



def assign_ranking_numbers(ordered_pred):
    ranknum = 0
    # take care of possible ties (especially for the sense baseline)
    ordered_rank_word_score = []
    for w, score in ordered_pred:
        if ordered_rank_word_score:
            if score != ordered_rank_word_score[-1][2]:
                ranknum +=1
        ordered_rank_word_score.append((ranknum, w, score))

    wordscores_by_rank = dict()
    for rank, w, score in ordered_rank_word_score:
        if rank not in wordscores_by_rank:
            wordscores_by_rank[rank] = []
        wordscores_by_rank[rank].append((w, score))

    return wordscores_by_rank


def make_freq_prediction(scale_words, freq_counts):
    freq_pred = ((w, freq_counts[w]) for w in scale_words)
    return freq_pred


def make_wordnet_prediction(scale_words, language="en", wordsenseinfo=[], dataset='demelo'):
    if language == "en":
        wordnet_pred = ((w, len(wn.synsets(w))) for w in scale_words)
    else:
        wordnet_pred = []
        av_wordsense_num = np.round(wordsenseinfo["avg-"+dataset])
        for w in scale_words:
            if w in wordsenseinfo:
                wordnet_pred.append((w, len(wordsenseinfo[w])))
            else:
                wordnet_pred.append((w, av_wordsense_num))

        wordnet_pred = tuple(wordnet_pred)
    return wordnet_pred


def make_staticdiffvec_prediction(scale_words, vectors, diffvec):
    static_distances_to_diff = ((w, cosine(vectors.query(w), diffvec)) for w in scale_words)
    return static_distances_to_diff

def make_diffvec_prediction(scale_words, sentence_data,  diffvecs, layer, X=10):
    vectors_per_word = defaultdict(list)

    if tuple(scale_words) not in sentence_data:
        return False

    for instance in sentence_data[tuple(scale_words)][:X]:
        for w in instance['representations']:
            vectors_per_word[w].append(instance['representations'][w][layer].numpy())

    avvector_per_word = {w: np.average(vectors_per_word[w], axis=0) for w in vectors_per_word}
    avvector_per_word2 = avvector_per_word

    distances_to_diff = ((w, cosine(avvector_per_word2[w], diffvecs[layer])) for w in avvector_per_word)

    return distances_to_diff


def make_bertdist_prediction(extreme_word, sentence_data, scale_words, X=10):
    for layer in range(1, 13):
        vectors_per_word = defaultdict(list)

        for instance in sentence_data[tuple(scale_words)][:X]:
            for w in instance['representations']:
                vectors_per_word[w].append(instance['representations'][w][layer].numpy())

        dists_to_extreme = defaultdict(list)
        for instance in sentence_data[tuple(scale_words)][:X]:
            for w in instance['representations']:
                dists_to_extreme[w].append(cosine(instance['representations'][w][layer].numpy(), instance['representations'][extreme_word][layer].numpy()))

        avg_toextreme = [(w, np.average(dists_to_extreme[w])) for w in dists_to_extreme]

    return avg_toextreme



def calculate_extreme_vector(scalar_sentence_data, method, language):
    reference_dict = dict()
    X = int(method.split("_")[-1][1:])
    method = method.split("_")[0]

    for l in ["en","fr","es","el"]:
        reference_dict[l] = dict()
    reference_dict["en"]["extremepos"] = ("good", "better","remarkable","exceptional", "perfect")
    reference_dict["es"]["extremepos"] = ("bueno", "perfecto")
    reference_dict["fr"]["extremepos"] = ("bon", "parfait")
    reference_dict["el"]["extremepos"] = ("καλός", "τέλειος")

    scale = reference_dict[language][method]
    extreme_word = scale[-1]

    extr_vectors_by_layer = dict()
    for layer in range(1, 13):
        extr_vectors_by_layer[layer] = []

    for instance in scalar_sentence_data[scale][:X]:
        for layer in range(1, 13):
            extreme_rep = instance['representations'][extreme_word][layer].numpy()
            extr_vectors_by_layer[layer].append(extreme_rep)

    final_extr_vector_by_layer = dict()
    for layer in extr_vectors_by_layer:
        final_extr_vector_by_layer[layer] = np.average(extr_vectors_by_layer[layer], axis=0)

    return final_extr_vector_by_layer



def calculate_static_extreme_vector(static_vectors, method, language):
    reference_dict = dict()
    for l in ["en", "fr", "es", "el"]:
        reference_dict[l] = dict()
    reference_dict["en"]["extremepos"] = ("good", "better","remarkable","exceptional", "perfect")
    reference_dict["es"]["extremepos"] = ("bueno", "perfecto")
    reference_dict["fr"]["extremepos"] = ("bon", "parfait")
    reference_dict["el"]["extremepos"] = ("καλός", "τέλειος")

    scale = reference_dict[language][method]
    extreme_word = scale[-1]

    return static_vectors.query(extreme_word)





