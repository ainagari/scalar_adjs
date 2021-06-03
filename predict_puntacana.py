import pickle
from operator import itemgetter
from collections import defaultdict
from pymagnitude import *
import argparse
from predict import *

def save_predicted_rankings(predicted_ranking_dict, args):
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    for dataname in predicted_ranking_dict:
        if not os.path.isdir(os.path.join(args.output_dir, dataname)):
            os.mkdir(os.path.join(args.output_dir, dataname))
        if not os.path.isdir(os.path.join(args.output_dir, dataname, args.sentences)):
            os.mkdir(os.path.join(args.output_dir, dataname, args.sentences))
        for scale in predicted_ranking_dict[dataname]:
            for method in predicted_ranking_dict[dataname][scale]:
                if not os.path.isdir(os.path.join(args.output_dir, dataname, args.sentences, method)):
                    os.mkdir(os.path.join(args.output_dir, dataname, args.sentences, method))

                newfilepath =os.path.join(args.output_dir, dataname, args.sentences, method, scale)
                with open(newfilepath, 'w') as out:
                    wordscores_by_rank = assign_ranking_numbers(predicted_ranking_dict[dataname][scale][method])
                    for rank in wordscores_by_rank:
                        ws = " || ".join([w for w, score in wordscores_by_rank[rank]])
                        scs = " || ".join([str(score) for w, score in wordscores_by_rank[rank]])
                        line = str(rank) + "\t" + ws + "\t" + scs + "\n"
                        out.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/", type=str,
                        help="Folder containing the scalar adjective datasets and the representations.")
    parser.add_argument("--sentences", default='ukwac', type=str,
                        help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'ukwac-random'")
    parser.add_argument("--methods", default='all', type=str,
                        help="methods to be used for prediction, separated with commas. Can be 'all'. All methods available are: "
                             "freq,wordnet,diffvec,staticdiffvec,bertdist,diffvec1scale,staticdiffvec1scale,diffvec1scaleneg, staticdiffvec1scaleneg, diffvec5scale, staticdiffvec5scale")
    parser.add_argument("--output_dir", default="predictions/", type=str,
                        help="The output directory where predictions will be written")
    parser.add_argument("--num_sentences", default=10, type=int,
                        help="Number of sentences to be used to build diffvec representations")
    parser.add_argument("--english_freq_fn", default="", type=str,
                        help="filename (including path) containing frequency counts. It should be a zipped text file where lines look like: word[TAB]freq\n")
    parser.add_argument("--path_to_static", default="", type=str,
                        help="The path to the directory containing magnitude fasttext embeddings")

    datanames = ['demelo', 'crowd', 'wilkinson']
    reference_dataset = "wilkinson"

    args = parser.parse_args()

    # LIST OF METHODS TO RUN
    if args.methods == 'all':
        methods = ['freq', 'wordnet', 'diffvec', 'staticdiffvec', 'bertdist', 'diffvec1scale', 'staticdiffvec1scale', "diffvec1scaleneg", "staticdiffvec1scaleneg", 'diffvec5scale', 'staticdiffvec5scale']
    else:
        methods = args.methods.split(",")

    #### LOAD RANKINGS
    rankings, adjs_by_dataset = load_rankings(args.data_dir, datanames=datanames)

    sentence_data = pickle.load(open(args.data_dir + "scalar_embeddings_" + args.sentences + ".pkl", "rb"))
    if "staticdiffvec" in methods or "staticdiffvec1scale" in methods or "staticdiffvec5scale" in methods or "staticdiffvec1scaleneg" in methods:
        static_vectors = Magnitude(args.path_to_static + "GoogleNews-vectors-negative300.magnitude")

    if 'freq' in methods:
        freq_counts = load_frequency(args.language)
    if 'diffvec' in methods:
        diffvecs_by_dataname = dict()
        for dataname in datanames:
            diffvecs_by_dataname.update(calculate_diff_vector(sentence_data, rankings[dataname], dataname, X=args.num_sentences))
    if 'staticdiffvec' in methods:
        static_diffvecs_by_dataname = dict()
        for dataname in datanames:
            static_diffvecs_by_dataname.update(calculate_static_diff_vector(static_vectors, rankings[dataname], dataname))

    diffvecs_singlescales = dict()
    staticdiffvecs_singlescales = dict()

    for method in methods:
        if method in ["diffvec1scale", "diffvec1scaleneg", "diffvec5scale"]:
            diffvecs_singlescales[method + "_X"+str(args.num_sentences)] = calculate_diff_vector_singlescales(sentence_data, method=method, X=args.num_sentences, reference_dataset=reference_dataset)
        elif method in ["staticdiffvec1scale", "staticdiffvec1scaleneg", "staticdiffvec5scale"]:
            staticdiffvecs_singlescales[method] = calculate_static_diffvector_singlescales(static_vectors, method=method, reference_dataset=reference_dataset,language=args.language)


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
                        pred = make_wordnet_prediction(scale_words, args.language)
                    # order predictions from mild to extreme
                    ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)  # the higher the freq/polysemy, the milder
                    predictions[dataname][scale][method] = ordered_pred

                elif method == "staticdiffvec":
                    for dataname_reference in datanames: # here dataname is the target (for which we make predictions) and dataname_reference is the dataset from which we built diffvec
                        if dataname_reference != dataname:
                            submethod = method + "-" + dataname_reference
                            pred = make_staticdiffvec_prediction(scale_words, static_vectors, static_diffvecs_by_dataname[dataname_reference + "-" + dataname]) # predictions are distances to diffvec
                            ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)  # the higher the distance to diffvec, the milder
                            predictions[dataname][scale][submethod] = ordered_pred
                elif method == "diffvec":
                    for dataname_reference in datanames:
                        if dataname_reference != dataname:
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
                        ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)  # the higher the distance, the milder
                        predictions[dataname][scale][submethod] = ordered_pred
                elif method == 'bertdist':
                    for layer in range(1,13):
                        submethod = "bertdist-" + method + "-extreme-" + str(layer)
                        extreme_word = scale_words[-1]
                        pred = make_bertdist_prediction(extreme_word, sentence_data, scale_words)
                        ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)
                        predictions[dataname][scale][submethod] = ordered_pred

    save_predicted_rankings(predictions, args)
