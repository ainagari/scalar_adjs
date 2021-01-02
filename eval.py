'''
Evaluation script for predictions obtained using predict.py
Adapted from the script at https://github.com/acocos/scalar-adj/blob/master/globalorder/src/eval.py
'''

import os, sys
import itertools
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import argparse
import pdb
from copy import deepcopy

import warnings
warnings.filterwarnings("error")


def read_rankings(dirname, apply_diffvec_ties=False):
    '''
    ::param: dirname : directory holding results files
    ::returns: dict  : {word: score}
    '''
    rankingfiles = os.listdir(dirname)
    results = {}
    mildextreme = dict()
    for rf in rankingfiles:
        results[rf] = {}
        mildextreme[rf] = dict()
        with open(os.path.join(dirname, rf), 'r') as fin:
            ordered_words = []
            if not apply_diffvec_ties:
                for line in fin:
                    score, words = line.strip().split('\t')[:2]
                    ordered_words.append(words)
                    words = words.split(' || ')
                    score = float(score)

                    for word in words:
                        results[rf][word] = score  # this is the ranking position
            elif apply_diffvec_ties:
                ordered_words_scores = []
                for line in fin:
                    score, words, actualscore = line.strip().split('\t')
                    ordered_words.append(words)
                    ordered_words_scores.append((words, float(actualscore)))

                w1, score1 = ordered_words_scores[0]
                results[rf][w1] = 0
                previous_score = score1
                rank = 0
                for w, score in ordered_words_scores[1:]:
                    if abs(score - previous_score) < 0.01: # a tie
                        results[rf][w] = rank
                    else:
                        rank += 1
                        previous_score = score
                        results[rf][w] = rank
            try:
                mildest_word = ordered_words[0].split(" || ")[0]
            except IndexError:
                pdb.set_trace()
            extreme_word = ordered_words[-1].split(" || ")[0]
            mildextreme[rf]['mild'] = mildest_word
            mildextreme[rf]['extreme'] = extreme_word

    return results, mildextreme



def compare_ranks(ranks):
    words = ranks.keys()
    pairs = [(i ,j) for i ,j in itertools.product(words, words) if i< j]
    compared = {}
    for p in pairs:
        w1, w2 = p
        if ranks[w1] < ranks[w2]:
            cls = "<"
        elif ranks[w1] > ranks[w2]:
            cls = ">"
        else:
            cls = "="
        compared[p] = cls
    return compared


def pairwise_accuracy(pred_rankings, gold_rankings, plustwo=False, omit_extreme_or_mild=None, gold_mildextreme=None):
    # plustwo means we only consider scales with +2 words (for BERTSIM, section 5.1 in the paper)
    results_by_pair = []
    g = []
    p = []
    for rankfile, predranks in pred_rankings.items():
        if plustwo:
            if len(predranks) <= 2:  # plustwo means we only take into account scales with +2 words
                continue
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Acc - Could not find gold rankings for cluster %s\n' % rankfile)
            continue

        goldranks = deepcopy(goldranks)
        predranks = deepcopy(predranks)
        if omit_extreme_or_mild:
            word_to_remove = gold_mildextreme[rankfile][omit_extreme_or_mild]
            del goldranks[word_to_remove]
            del predranks[word_to_remove]

        gold_compare = compare_ranks(goldranks)
        pred_compare = compare_ranks(predranks)
        gold_compare = deepcopy(gold_compare)
        pred_compare = deepcopy(pred_compare)

        if not gold_compare:
            continue

        if set(gold_compare.keys()) != set(pred_compare.keys()):
            sys.stderr.write('Acc - ERROR: Key mismatch for cluster %s\n' % rankfile)
            continue
        pairs = sorted(list(gold_compare.keys()))
        for pr in pairs:
            g.append(gold_compare[pr])
            p.append(pred_compare[pr])
            if gold_compare[pr] == pred_compare[pr]:
                results_by_pair.append((pr, gold_compare[pr], pred_compare[pr], 1))
            else:
                results_by_pair.append((pr, gold_compare[pr], pred_compare[pr], 0))


    correct = [1. if gg == pp else 0. for gg, pp in zip(g, p)]

    acc = sum(correct) / len(correct)

    return acc, results_by_pair


def kendalls_tau(pred_rankings, gold_rankings, plustwo=False, omit_extreme_or_mild=None,gold_mildextreme=None):
    # plustwo means we only consider scales with +2 words (for BERTSIM, section 5.1 in the paper)
    results_by_scale = dict()
    taus = []
    ns = []
    for rankfile, predranks in pred_rankings.items():
        if plustwo:
            if len(predranks) <= 2:
                continue
        n_c = 0.
        n_d = 0.
        n = 0.
        ties_g = 0.
        ties_p = 0.
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Kendall - Could not find gold rankings for cluster %s\n' % rankfile)
            continue
        goldranks = deepcopy(goldranks)
        predranks = deepcopy(predranks)
        if omit_extreme_or_mild:
            word_to_remove = gold_mildextreme[rankfile][omit_extreme_or_mild]
            del goldranks[word_to_remove]
            del predranks[word_to_remove]

        words = sorted(goldranks.keys())
        for w_i in words:
            for w_j in words:
                if w_j >= w_i:
                    continue
                n += 1
                # check ties
                tied = False
                if goldranks[w_i] == goldranks[w_j]:
                    ties_g += 1
                    tied = True
                if predranks[w_i] == predranks[w_j]:
                    ties_p += 1
                    tied = True
                if tied:
                    continue
                # concordant/discordant
                dir_g = np.sign(goldranks[w_j] - goldranks[w_i])
                dir_p = np.sign(predranks[w_j] - predranks[w_i])
                if dir_g == dir_p:
                    n_c += 1
                else:
                    n_d += 1
        try:
            tau = (n_c - n_d) / np.sqrt((n - ties_g) * (n - ties_p))
        except RuntimeWarning:
            tau = 0.

        results_by_scale[rankfile] = (words, tau)
        taus.append(tau)
        ns.append(n)

    taus = [t if not np.isnan(t) else 0. for t in taus]
    tau_avg = np.average(taus, weights=ns)

    return tau_avg, results_by_scale


def spearmans_rho_avg(pred_rankings, gold_rankings, plustwo=True, omit_extreme_or_mild=None, gold_mildextreme=None):
    # plustwo means we only take into account scales with +2 words
    results_by_scale = dict()
    rhos = []
    ns = []
    for rankfile, predranks in pred_rankings.items():
        if plustwo:
            if len(predranks) <= 2:
                continue
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Spearman -Could not find gold rankings for cluster %s\n' % rankfile)
            continue

        if set(goldranks.keys()) != set(predranks.keys()):
            pdb.set_trace()
            sys.stderr.write('Spearman -ERROR: Key mismatch for cluster %s\n' % rankfile)
            continue

        goldranks = deepcopy(goldranks)
        predranks = deepcopy(predranks)

        if omit_extreme_or_mild:
            word_to_remove = gold_mildextreme[rankfile][omit_extreme_or_mild]
            del goldranks[word_to_remove]
            del predranks[word_to_remove]

        ns.append(len(goldranks))
        words = sorted(goldranks.keys())

        predscores = [predranks[w] for w in words]
        goldscores = [goldranks[w] for w in words]

        try:
            r, p = spearmanr(predscores, goldscores)
        except RuntimeWarning:
            r = 0.
        if np.isnan(r):
            r = 0.
        rhos.append(r)
        results_by_scale[rankfile] = (words, r)

    rho = np.average(rhos, weights=ns)

    return rho, results_by_scale




def save_results(results_dict, args):
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for dataname in results_dict:
        if not os.path.isdir(os.path.join(output_dir, dataname)):
            os.mkdir(os.path.join(output_dir, dataname))
        for sentence_type in results_dict[dataname]:

            if not os.path.isdir(os.path.join(output_dir, dataname, sentence_type)):
                os.mkdir(os.path.join(output_dir, dataname, sentence_type))

            elif args.apply_diffvec_ties:
                newfilename = os.path.join(output_dir, dataname, sentence_type) + "/results_diffvecties.csv"
            else:
                newfilename = os.path.join(output_dir, dataname, sentence_type) + "/results.csv"

            df = pd.DataFrame.from_dict(results_dict[dataname][sentence_type], orient='index')

            df.to_csv(newfilename, sep="\t")



def calculate_coverage(pred_rankings, gold_rankings):
    total_len = len(gold_rankings)
    coverage = 0
    for rf, _ in gold_rankings.items():
        if rf in pred_rankings:
            coverage += 1
    coverage = coverage / total_len

    return coverage



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/", type=str,
                        help="Folder containing the scalar adjective datasets.")
    parser.add_argument("--pred_dir", default="predictions/", type=str,
                        help="Folder containing the predictions for the scalar adjective datasets")
    parser.add_argument("--sentences", default='all', type=str,
                        help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'random-ukwac'. For languages other than english it is automatically oscar.")
    parser.add_argument("--apply_diffvec_ties", action="store_true",
                        help="whether we help the diffvec models propose ties")
    parser.add_argument("--output_dir", default="results/", type=str,
                        help="Folder to store the evaluation results.")


    args = parser.parse_args()



    datanames = ['demelo', 'crowd', 'wilkinson']
    if args.sentences == "all":
        sentences = ['ukwac', 'flickr', 'ukwac-random']
    else:
        sentences = args.sentences.split(",")

    pred_dir = args.pred_dir

    results = dict()
    results_by_scale = dict()

    ####### EVALUATION LOOP
    for dataname in datanames:
        print(dataname)
        gold_rankings, gold_mildextreme = read_rankings(
            args.data_dir + dataname + "/gold_rankings/")  # load gold rankings
        results[dataname] = dict()
        results_by_scale[dataname] = dict()
        for sentence_type in sentences:
            diri = os.path.join(pred_dir, dataname, sentence_type)
            results[dataname][sentence_type] = dict()
            results_by_scale[dataname][sentence_type] = dict()

        for method in os.listdir(diri):  # consider all methods found in that directory
            results[dataname][sentence_type][method] = dict()
            results_by_scale[dataname][sentence_type][method] = dict()
            # read predictions, modify them accounting for ties if necessary
            if args.apply_diffvec_ties and "diffvec" in method:
                pred_rankings, _ = read_rankings(os.path.join(diri, method), apply_diffvec_ties=True)
                method += "-01ties"
            elif not args.apply_diffvec_ties:
                pred_rankings, _ = read_rankings(os.path.join(diri, method))

            ###### First: evaluating on scales with +2 words (only bertdist [bertsim], freq and wordnet)
            if "bertdist" in method or method in ["freq","wordnet"]:
                results[dataname][sentence_type][method]['pairwise_accuracy_+2'], results_by_scale[dataname][sentence_type][method]['pairwise_accuracy_+2'] = pairwise_accuracy(pred_rankings, gold_rankings, plustwo=True, omit_extreme_or_mild="extreme", gold_mildextreme=gold_mildextreme)
                results[dataname][sentence_type][method]["kendalls_tau_+2"], results_by_scale[dataname][sentence_type][method]['kendalls_tau_+2'] = kendalls_tau(pred_rankings,gold_rankings,plustwo=True,omit_extreme_or_mild="extreme",gold_mildextreme=gold_mildextreme)
                results[dataname][sentence_type][method]["spearmans_rho_avg_+2"], results_by_scale[dataname][sentence_type][method]['spearmans_rho_avg_+2'] = spearmans_rho_avg(pred_rankings, gold_rankings, plustwo=True, omit_extreme_or_mild="extreme", gold_mildextreme=gold_mildextreme)


            if "bertdist" not in method: #freq, wordnet, and all diffvec methods
                results[dataname][sentence_type][method]['pairwise_accuracy'], results_by_scale[dataname][sentence_type][method]['pairwise_accuracy'] = pairwise_accuracy(pred_rankings,gold_rankings)
                results[dataname][sentence_type][method]["kendalls_tau"], results_by_scale[dataname][sentence_type][method]['kendalls_tau'] = kendalls_tau(pred_rankings, gold_rankings)
                results[dataname][sentence_type][method]["spearmans_rho_avg"], results_by_scale[dataname][sentence_type][method]['spearmans_rho_avg'] = spearmans_rho_avg(pred_rankings, gold_rankings)

                ## Coverage
                coverage = calculate_coverage(pred_rankings, gold_rankings)
                results[dataname][sentence_type][method]["coverage"] = coverage


    save_results(results, args)


