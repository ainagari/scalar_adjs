'''
Different methods to make ranking predictions for the scalar adjective datasets
'''

from read_scalar_datasets import read_scales
from nltk.corpus import wordnet as wn
import gzip
import pickle
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



def assign_reference_scales(reference_dataset, method):
    if reference_dataset == "wilkinson":
        if method in ["diffvec1scale", "staticdiffvec1scale"]:
            reference_scales = [('good', 'great', 'wonderful', 'awesome')]
        elif method in ["diffvec1scaleneg","staticdiffvec1scaleneg"]:
            reference_scales = [('bad', 'awful', 'terrible', 'horrible')]
        elif method in ["diffvec5scale","staticdiffvec5scale"]:
            reference_scales = [('good', 'great', 'wonderful', 'awesome'), ('bad', 'awful', 'terrible', 'horrible'),
                                ('old', 'ancient'), ('pretty', 'beautiful', 'gorgeous'), ('ugly', 'hideous')]
    return reference_scales



def calculate_static_diffvector_singlescales(vectors, method="staticdiffvec1scale", reference_dataset="wilkinson"):
    reference_scales = assign_reference_scales(reference_dataset=reference_dataset, method=method)

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



def calculate_diff_vector_singlescales(data, method="diffvec1scale", X=10, reference_dataset="wilkinson"):
    reference_scales = assign_reference_scales(reference_dataset=reference_dataset, method=method)

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

def load_frequency(freq_path):
    freq_counts = defaultdict(int)
    frequency_fn = freq_path

    with gzip.open(frequency_fn) as f:
        bytecontents = f.read()
    contents = bytecontents.decode("utf-8")
    contents = contents.split("\n")

    for tokencount in contents:
        s = tokencount.strip().split("\t")
        if len(s) == 2:
            token, count = s
            freq_counts[token] = int(count)

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

def make_wordnet_prediction(scale_words):
    wordnet_pred = ((w, len(wn.synsets(w))) for w in scale_words)
    return wordnet_pred

def make_wordnetpos_prediction(scale_words):
    wordnetpos_pred = []
    for w in scale_words:
        num = len(wn.synsets(w, 'a'))
        if num > 0:
            wordnetpos_pred.append((w, num))
        elif num == 0:
            wordnetpos_pred.append((w, len(wn.synsets(w))))
    return wordnetpos_pred


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

    distances_to_diff = ((w, cosine(avvector_per_word2[w], diffvecs[layer])) for w in
                         avvector_per_word)
    return distances_to_diff

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
                        line = str(rank) + "\t" + ws  + "\t" + scs + "\n"
                        out.write(line)


def make_bertdist_prediction(extreme_word, sentence_data, scale_words, X=10):
    for layer in range(1, 13):
        vectors_per_word = defaultdict(list)

        for instance in sentence_data[tuple(scale_words)][:X]:
            for w in instance['representations']:
                vectors_per_word[w].append(instance['representations'][w][layer].numpy())

        # get the distances, then average them
        dists_to_extreme = defaultdict(list)
        for instance in sentence_data[tuple(scale_words)][:X]:
            for w in instance['representations']:
                dists_to_extreme[w].append(cosine(instance['representations'][w][layer].numpy(), instance['representations'][extreme_word][layer].numpy()))

        avg_toextreme = [(w, np.average(dists_to_extreme[w])) for w in dists_to_extreme]

    return avg_toextreme




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/", type=str,
                        help="Folder containing the scalar adjective datasets.")
    parser.add_argument("--sentences", default='ukwac', type=str,
                        help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'ukwac-random'")
    parser.add_argument("--methods", default='all', type=str,
                        help="methods to be used for prediction, separated with commas. Can be 'all'. All methods available are: "
                             "freq,wordnet,diffvec,staticdiffvec,bertdist,diffvec1scale,staticdiffvec1scale,diffvec1scaleneg, staticdiffvec1scaleneg, diffvec5scale, staticdiffvec5scale")
    parser.add_argument("--output_dir", default="predictions/", type=str,
                        help="The output directory where predictions will be written")
    parser.add_argument("--num_sentences", default=10, type=int,
                        help="Number of sentences to be used to build diffvec representations")
    parser.add_argument("--freq_file", type=str,
                        help="filename (including path) containing frequency counts. It should be a zipped text file where lines look like: word[TAB]freq\n")


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
        static_vectors = Magnitude(args.data_dir + "GoogleNews-vectors-negative300.magnitude")


    if 'freq' in methods:
        freq_counts = load_frequency(args.freq_file)
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
        if method in ["diffvec1scale", "diffvec1scaleneg","diffvec5scale"]:
            diffvecs_singlescales[method + "_X"+str(args.num_sentences)] = calculate_diff_vector_singlescales(sentence_data, method=method, X=args.num_sentences, reference_dataset=reference_dataset)
        elif method in ["staticdiffvec1scale","staticdiffvec1scaleneg","staticdiffvec5scale"]:
            staticdiffvecs_singlescales[method] = calculate_static_diffvector_singlescales(static_vectors, method=method, reference_dataset=reference_dataset)


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
                        pred = make_wordnet_prediction(scale_words)
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







