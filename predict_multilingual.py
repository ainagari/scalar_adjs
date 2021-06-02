
import gzip
import pickle
import numpy as np
import sys
from scipy.spatial.distance import cosine
from operator import itemgetter
from pymagnitude import *
import argparse
from predict import *


def save_predicted_rankings(predicted_ranking_dict, args):
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str)):
        os.mkdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str))
    for dataname in predicted_ranking_dict:
        if not os.path.isdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str, dataname)):
            os.mkdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str, dataname))
        if not os.path.isdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str, dataname, args.sentences)):
            os.mkdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str, dataname, args.sentences))
        for scale in predicted_ranking_dict[dataname]:
            for method in predicted_ranking_dict[dataname][scale]:
                method_str = method
                if not os.path.isdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str, dataname, args.sentences, method_str)):
                    os.mkdir(os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str, dataname, args.sentences, method_str))
                newfilepath = os.path.join(args.output_dir, args.language_str + "_" + args.bpe_str, dataname, args.sentences, method_str, scale)

                with open(newfilepath, 'w') as out:
                    wordscores_by_rank = assign_ranking_numbers(predicted_ranking_dict[dataname][scale][method])
                    for rank in wordscores_by_rank:
                        ws = " || ".join([w for w, score in wordscores_by_rank[rank]])
                        scs = " || ".join([str(score) for w, score in wordscores_by_rank[rank]])
                        line = str(rank) + "\t" + ws + "\t" + scs + "\n"
                        out.write(line)



if __name__ == "__main__":

    datanames = ['demelo', 'crowd', 'wilkinson']

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="multilingual_data/", type=str,
                        help="Folder containing the scalar adjective datasets, representations, and additional data (frequencies and #of senses). For English, it should be data/")
    parser.add_argument("--sentences", default='ukwac', type=str,
                        help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'ukwac-random'")
    parser.add_argument("--methods", default='all', type=str,
                        help="methods to be used for prediction, separated with commas. Can be 'all'. All methods available are: freq, wordnet, "
                             "diffvec, staticdiffvec, bertdist, diffvec1scale, staticdiffvec1scale, diffvec1scaleneg, staticdiffvec1scaleneg, "
                             "diffvec5scale, staticdiffvec5scale")
    parser.add_argument("--output_dir", default="multilingual_predictions/", type=str,
                        help="The output directory where predictions will be written")
    parser.add_argument("--num_sentences", default=10, type=int,
                        help="Number of sentences to be used to build diffvec representations")
    parser.add_argument("--english_freq_fn", default="", type=str,
                        help="filename (including path) containing frequency counts. It should be a zipped text file where lines look like: word[TAB]freq\n")
    parser.add_argument("--language", default="en", type=str,
                        help="What language we are working with: en, es, el, fr.")
    parser.add_argument("--multilingual-uncased", action="store_true",
                        help="Whether we use multilingual BERT. Otherwise the monolingual BERT of the chosen language will be used.")
    parser.add_argument("--multilingual-cased", action="store_true",
                        help="Whether we use multilingual BERT. Otherwise the monolingual BERT of the chosen language will be used.")
    parser.add_argument("--exclude_last_bpe", action="store_true", type=str, help="whether we exclude the last piece when a word is split into multiple wordpieces."
                                                                                  "Otherwise, we use the representations of all pieces.")
    parser.add_argument("--path_to_static", default="", type=str, help="The path to the directory containing magnitude fasttext embeddings")
    args = parser.parse_args()

    reference_dataset = "crowd"
    print("The reference dataset is", reference_dataset)

    if args.methods == 'all':
        methods = ['freq', 'wordnet', 'diffvec', 'staticdiffvec', 'bertdist', 'diffvec1scale', 'staticdiffvec1scale',
                   "diffvec1scaleneg", "staticdiffvec1scaleneg", 'diffvec5scale', 'staticdiffvec5scale']
    else:
        methods = args.methods.split(",")

    if args.multilingual-uncased:
        args.language_str = "multi-" + args.language
    elif args.multilingual-cased:
        args.language_str = "multicased-" + args.language
    else:
        args.language_str = args.language

    if not args.exclude_last_bpe:
        args.bpe_str = "all-bpes"
    else:
        args.bpe_str = "exclude-last-bpes"

    if args.language in ["es","el","fr"]:
        datanames = ["wilkinson", "demelo"]
        args.sentences = "oscar"
        if args.language == "el" and "wordnet" in methods:
            methods.remove("wordnet")
        elif args.language != "el":
            wordsenseinfo = pickle.load(open(args.data_dir + "wordsense_info_concept_" + args.language + ".pkl", "rb"))
    elif args.language == "en":
        wordsenseinfo = []
    sentence_data = pickle.load(open(args.data_dir + "/scalar_embeddings_" + args.language_str + "_" + args.bpe_str + "_" + args.sentences + ".pkl", "rb"))

    if args.multilingual-uncased or args.multilingual-cased:
        print("There are no multilingual static vectors for now")
        methods_to_remove = [x for x in methods if "static" in methods]
        for mtr in methods_to_remove:
            methods.remove(mtr)

    vectors = Magnitude(args.path_to_static + "cc." + args.language + ".300.magnitude")

    #### LOAD RANKINGS
    rankings, adjs_by_dataset = load_rankings(args.data_dir + args.language + "/", datanames=datanames)
    # load info about the number of senses
    for d in adjs_by_dataset:
        senses_acc = []
        for a in adjs_by_dataset[d]:
            if a in wordsenseinfo:
                senses_acc.append(len(wordsenseinfo[a]))
        wordsenseinfo["avg-"+d] = np.average(senses_acc)


    if 'freq' in methods:
        freq_counts = load_frequency(args.language, args.english_frequency_fn)
    if 'diffvec' in methods:
        diffvecs_by_dataname = dict()
        for dataname in datanames:
            diffvecs_by_dataname.update(calculate_diff_vector(sentence_data, rankings[dataname], dataname, X=args.num_sentences))
    if 'staticdiffvec' in methods:
        static_diffvecs_by_dataname = dict()
        for dataname in datanames:
            static_diffvecs_by_dataname.update(calculate_static_diff_vector(vectors, rankings[dataname], dataname))

    diffvecs_singlescales = dict()
    staticdiffvecs_singlescales = dict()


    for method in methods:
        if method in ["diffvec1scale", "diffvec1scaleneg"]:
            diffvecs_singlescales[method + "_X" + str(args.num_sentences)] = calculate_diff_vector_singlescales(sentence_data, method=method, X=args.num_sentences, reference_dataset=reference_dataset, language=args.language)
        elif method in ["staticdiffvec1scale", "staticdiffvec1scaleneg"]:
            staticdiffvecs_singlescales[method] = calculate_static_diffvector_singlescales(static_vectors, method=method, reference_dataset=reference_dataset, language=args.language)

    # PREDICTION LOOP
    predictions = dict()
    for dataname in rankings:
        print(dataname)
        predictions[dataname] = dict()
        for scale in rankings[dataname]:
            predictions[dataname][scale] = dict()
            scale_words = get_scale_words(rankings[dataname][scale])
            for method in methods:
                if method in ['freq', 'wordnet']:
                    if method == 'freq':
                        pred = make_freq_prediction(scale_words, freq_counts)
                    elif method == 'wordnet':
                        pred = make_wordnet_prediction(scale_words, language=args.language, wordsenseinfo=wordsenseinfo, dataset=dataname)
                    ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)   # the higher the freq/polysemy, the milder
                    predictions[dataname][scale][method] = ordered_pred

                elif method == "staticdiffvec":
                    for dataname_reference in datanames: # here dataname is the target (for which we make predictions) and dataname_reference is the dataset from which we built diffvec
                        if dataname_reference != dataname:
                            submethod = method + "-" + dataname_reference
                            pred = make_staticdiffvec_prediction(scale_words, vectors, static_diffvecs_by_dataname[dataname_reference + "-" + dataname]) # predictions are distances to diffvec
                            ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)  # the higher the distance to diffvec, the milder
                            predictions[dataname][scale][submethod] = ordered_pred
                elif method == "diffvec":
                    for dataname_reference in datanames:
                        if dataname_reference != dataname:  # don't use wilkinson vectors to make predictions on wilkinson!
                            for layer in range(1, 13):
                                submethod = method + "-" + str(layer) + '-' + dataname_reference
                                pred = make_diffvec_prediction(scale_words, sentence_data, diffvecs_by_dataname[dataname_reference + "-" + dataname], layer)
                                ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)  # the higher the distance, the milder
                                predictions[dataname][scale][submethod] = ordered_pred
                elif method in staticdiffvecs_singlescales.keys():
                    pred = make_staticdiffvec_prediction(scale_words, static_vectors, staticdiffvecs_singlescales[method])
                    ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)
                    predictions[dataname][scale][method] = ordered_pred
                elif method in diffvecs_singlescales.keys():
                    for layer in range(1, 13):
                        submethod = method + "-" + str(layer)
                        pred = make_diffvec_prediction(scale_words, sentence_data, diffvecs_singlescales[method], layer, X=args.num_sentences)
                        ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)
                        predictions[dataname][scale][submethod] = ordered_pred
                elif method == 'bertdist':
                    for layer in range(1, 13):
                        submethod = "bertdist-" + method + "-extreme-" + str(layer)
                        extreme_word = scale_words[-1]
                        pred = make_bertdist_prediction(extreme_word, sentence_data, scale_words)
                        ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)
                        predictions[dataname][scale][submethod] = ordered_pred

    save_predicted_rankings(predictions, args)



