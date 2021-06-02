
import os
import argparse

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