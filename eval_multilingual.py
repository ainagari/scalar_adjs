
import os
import argparse
from eval import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="multilingual_data/", type=str,
                        help="Folder containing the scalar adjective datasets.")
    parser.add_argument("--pred_dir", default="multilingual_predictions/", type=str,
                        help="Folder containing the predictions for the scalar adjective datasets. Not necessary if we specify the language.")
    parser.add_argument("--sentences", default='all', type=str,
                        help="set of sentences used for methods that require sentences. Can be 'ukwac', 'flickr', or 'random-ukwac'. For languages other than english it is automatically oscar.")
    parser.add_argument("--apply_diffvec_ties", action="store_true",
                        help="whether we help our diffvec models propose ties")
    parser.add_argument("--output_dir", default="multilingual_results/", type=str,
                        help="Folder to store the evaluation results.")
    parser.add_argument("--language", default="en", type=str,
                        help="What language we are working with: en, es, el, fr.")
    parser.add_argument("--multilingual-uncased", action="store_true",
                        help="Whether we use multilingual BERT uncased. If no multilingual model is chosen, the monolingual BERT of the chosen language will be used.")
    parser.add_argument("--multilingual-cased", action="store_true",
                        help="Whether we use multilingual BERT cased. If no multilingual model is chosen, the monolingual BERT of the chosen language will be used.")    
    parser.add_argument("--exclude_last_bpe", action="store_true", type=str, help="whether we exclude the last piece when a word is split into multiple wordpieces."
                                                                                  "Otherwise, we use the representations of all pieces.")
    args = parser.parse_args()

    

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

    if args.language in ["es", "el", "fr"]:
        args.data_dir = args.data_dir + args.language + "/"
        datanames = ["wilkinson", "demelo"]
        args.sentences = "oscar"
        

    elif args.language == "en":
        datanames = ['demelo', 'crowd', 'wilkinson']
        if args.sentences == "all":
        args.sentences = ['ukwac', 'flickr', 'ukwac-random']
    else:
        args.sentences = args.sentences.split(",")

    
    pred_dir = args.pred_dir

    results = dict()    

    ####### EVALUATION LOOP
    for dataname in datanames:
        print(dataname)
        gold_rankings, gold_mildextreme = read_rankings(
            args.data_dir + dataname + "/gold_rankings/")  # load gold rankings
        results[dataname] = dict() 
        
        for sentence_type in sentences:
            diri = os.path.join(pred_dir, dataname, sentence_type)
            results[dataname][sentence_type] = dict()            
      
        for method in os.listdir(diri):  # consider all methods found in that directory                
            results[dataname][sentence_type][method] = dict()

            results[dataname][sentence_type][method]['pairwise_accuracy'], results_by_scale[dataname][sentence_type][method]['pairwise_accuracy'] = pairwise_accuracy(pred_rankings,gold_rankings)
            results[dataname][sentence_type][method]["kendalls_tau"], results_by_scale[dataname][sentence_type][method]['kendalls_tau'] = kendalls_tau(pred_rankings, gold_rankings)                                                                                   
            
            rho, d = spearmans_rho_avg(pred_rankings, gold_rankings)
            results[dataname][sentence_type][method]["spearmans_rho_avg"] = rho            

            acc_mild, acc_extreme = mildest_extreme_accuracy(pred_rankings, gold_rankings)
            results[dataname][sentence_type][method]["mildest_accuracy"] = acc_mild
            results[dataname][sentence_type][method]["extreme_accuracy"] = acc_extreme            
            
            
    save_results(results, args)


