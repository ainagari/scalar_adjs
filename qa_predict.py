'''
Make predictions for the Indirect QA task and evaluate them
'''

from predict import *
import pdb
import pickle
from scipy.spatial.distance import cosine
from operator import itemgetter
from pymagnitude import *



def prf_macro(y, yhat):
    yclean, yhatclean = zip(*[(yy, yh) for yy, yh in zip(y, yhat) if yy != 'uncertain'])
    classes = set(yclean)
    ps = []
    rs = []
    for cls in classes:
        p, r, f = prf(yclean, yhatclean, cls)
        ps.append(p)
        rs.append(r)
    n = len(classes)
    pavg = sum(ps) / n
    ravg = sum(rs) / n
    favg = (2 * pavg * ravg) / (pavg + ravg)
    return pavg, ravg, favg


def prf(y, yhat, cls):
    ymod = [yy == cls for yy in y]
    yhatmod = [yy == cls for yy in yhat]
    correct = [yy * yh for yy, yh in zip(ymod, yhatmod)]
    try:
        p = float(sum(correct)) / sum(yhatmod)
    except ZeroDivisionError:
        p = 0.
    try:
        r = float(sum(correct)) / sum(ymod)
    except ZeroDivisionError:
        r = 0.
    try:
        f = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f = 0.
    return p, r, f


def acc(y, yhat):
    corr = [float(yy == yh) for yy, yh in zip(y, yhat)]
    return sum(corr) / len(corr)


def make_diffvec_qa_prediction(qa_instance,  diffvecs, layer):

    question_rep = qa_instance["representations1"][layer]
    answer_rep = qa_instance["representations2"][layer]
    distances_to_diff = []
    distances_to_diff.append((qa_instance["AdjectiveA"].lower(), cosine(question_rep, diffvecs[layer])))
    distances_to_diff.append((qa_instance["AdjectiveB"].lower(), cosine(answer_rep, diffvecs[layer])))

    return distances_to_diff



def reason_prediction(qainstance, prediction, diffvec_tie_adjustment=False):
    '''The prediction is basically the two adjectives ordered from milder to extreme'''
    adjA = qainstance['AdjectiveA'].lower()
    adjB = qainstance['AdjectiveB'].lower()

    milder, score_m = prediction[0]
    extreme, score_x = prediction[1]
    if score_m == score_x: # this would be a tie
        answer = 'yes'
    if adjA == milder and adjB == extreme:
        answer = 'yes'
    elif adjA == extreme and adjB == milder:
        answer = 'no'
    if diffvec_tie_adjustment and abs(score_m - score_x) < 0.01:
        answer = 'yes'

    if str(qainstance['Negation']) != "nan": # if there is a negation
        if answer == 'yes':
            answer = 'no'
        elif answer == 'no':
            answer = 'yes'

    return answer



def adapt_adjectives_for_frequency(adj):
    if "/" in adj:
        adj = adj.split("/")[0]
    if adj in ["enough to be leader", "better than nothing", "worried to death"]:
        adj = adj.split()[0]
    elif adj in ["in trouble", "all right"]:
        adj = adj.split()[-1]
    return adj


if __name__ == "__main__":
    methods = ['freq', 'wordnet', 'diffvec', 'staticdiffvec', 'diffvec1scale', 'staticdiffvec1scale']
    qa_data = pickle.load(open("QA_instances_representations.pkl", "rb"))
    datanames = ['demelo','crowd','wilkinson']
    sentence_type = "ukwac"
    print("loaded qa data")

    results = dict()
    static_vectors = Magnitude("data/GoogleNews-vectors-negative300.magnitude")
    freq_counts = load_frequency()

    sentence_data = pickle.load(open("scalar_embeddings_" + sentence_type + ".pkl", "rb"))
    rankings = load_rankings("data/")

    if 'diffvec' in methods:
        diffvecs_by_dataname = dict()
        for dataname in datanames:
            diffvecs_by_dataname.update(
                calculate_diff_vector(sentence_data, rankings[dataname], dataname, X=10))
    if 'staticdiffvec' in methods:
        static_diffvecs_by_dataname = dict()
        for dataname in datanames:
            static_diffvecs_by_dataname.update(
                calculate_static_diff_vector(static_vectors, rankings[dataname], dataname))

    diffvecs_singlescales = dict()
    staticdiffvecs_singlescales = dict()

    for method in methods:
        if method in ["diffvec1scale", "diffvec1scaleneg", "diffvec5scale"]:
            diffvecs_singlescales[method + "_X10"] = calculate_diff_vector_singlescales(sentence_data, method=method, X=10, reference_dataset="wilkinson")
        elif method in ["staticdiffvec1scale", "staticdiffvec1scaleneg", "staticdiffvec5scale"]:
            staticdiffvecs_singlescales[method] = calculate_static_diffvector_singlescales(static_vectors, method=method, reference_dataset="wilkinson")



    # PREDICTION LOOP
    predictions = dict()  # idi is HITId, then there's method : prediction (yes, no, uncertain)
    predictions['gold'] = dict()

    j = 0
    for instance in qa_data:
        j+=1
        if instance['TriDominantAnswer'] == "uncertain":
            continue

        adjA = instance['AdjectiveA'].lower()
        adjB = instance['AdjectiveB'].lower()

        idi = instance['HITId']
        predictions['gold'][idi] = instance['TriDominantAnswer']
        for method in methods:
            if method in ['freq','wordnet']:
                if method == 'freq':
                    adjA_adapted = adapt_adjectives_for_frequency(adjA)
                    adjB_adapted = adapt_adjectives_for_frequency(adjB)

                    almost_unordered_adjs = make_freq_prediction([adjA_adapted, adjB_adapted], freq_counts)
                    unordered_adjs = []
                    for w, score in almost_unordered_adjs:
                        if w == adjA_adapted:
                            unordered_adjs.append((adjA, score))
                        elif w == adjB_adapted:
                            unordered_adjs.append((adjB, score))
                        else:
                            pdb.set_trace()

                elif method == 'wordnet':
                    unordered_adjs = make_wordnet_prediction([adjA, adjB])

                ordered_adjs = sorted(unordered_adjs, key=itemgetter(1), reverse=True)  # the higher the prob of being polysemous, the milder # from mild to extreme

                prediction = reason_prediction(instance, ordered_adjs)
                if method not in predictions:
                    predictions[method] = dict()

                predictions[method][idi] = prediction



            elif method == "staticdiffvec":
                for datacombi in static_diffvecs_by_dataname:
                    sourcedataset = datacombi.split("-")[0]
                    submethod = method + "-" + sourcedataset
                    if submethod in predictions and idi in predictions[submethod]:
                        continue

                    unordered_adjs = make_staticdiffvec_prediction([adjA, adjB], static_vectors, static_diffvecs_by_dataname[datacombi])
                    ordered_adjs = sorted(unordered_adjs, key=itemgetter(1), reverse=True)  # the higher the distance, the milder

                    prediction = reason_prediction(instance, ordered_adjs)
                    prediction_ties = reason_prediction(instance, ordered_adjs, diffvec_tie_adjustment=True)

                    if submethod not in predictions:
                        predictions[submethod] = dict()
                    if submethod + "-01ties" not in predictions:
                        predictions[submethod + "-01ties"] = dict()
                    predictions[submethod][idi] = prediction
                    predictions[submethod + "-01ties"][idi] = prediction_ties



            elif method == "diffvec":
                for datacombi in diffvecs_by_dataname:
                    sourcedataset = datacombi.split("-")[0]
                    for layer in range(1, 13):
                        submethod = method + '-' + str(layer) + '-' + sourcedataset
                        if submethod in predictions and idi in predictions[submethod]:
                            continue
                        unordered_adjs = make_diffvec_qa_prediction(instance, diffvecs_by_dataname[datacombi], layer)
                        ordered_adjs = sorted(unordered_adjs, key=itemgetter(1), reverse=True)
                        prediction = reason_prediction(instance, ordered_adjs)
                        prediction_ties = reason_prediction(instance, ordered_adjs, diffvec_tie_adjustment=True)
                        if submethod not in predictions:
                            predictions[submethod] = dict()
                        if submethod + "-01ties" not in predictions:
                            predictions[submethod + "-01ties"] = dict()
                        predictions[submethod][idi] = prediction
                        predictions[submethod + "-01ties"][idi] = prediction_ties

            elif method in staticdiffvecs_singlescales.keys():
                unordered_adjs = make_staticdiffvec_prediction([adjA, adjB], static_vectors, staticdiffvecs_singlescales[method])
                ordered_adjs = sorted(unordered_adjs, key=itemgetter(1),reverse=True)  # the higher the distance, the milder
                prediction = reason_prediction(instance, ordered_adjs)
                prediction_ties = reason_prediction(instance, ordered_adjs, diffvec_tie_adjustment=True)
                if method not in predictions:
                    predictions[method] = dict()
                if method + "-01ties" not in predictions:
                    predictions[method + "-01ties"] = dict()
                predictions[method][idi] = prediction
                predictions[method + "-01ties"][idi] = prediction_ties

            elif method in diffvecs_singlescales.keys():
                for layer in range(1, 13):
                    submethod = method + "-" + str(layer)
                    unordered_adjs = make_diffvec_qa_prediction(instance, diffvecs_singlescales[method], layer)
                    ordered_adjs = sorted(unordered_adjs, key=itemgetter(1), reverse=True)  # the higher the distance, the milder
                    prediction = reason_prediction(instance, ordered_adjs)
                    prediction_ties = reason_prediction(instance, ordered_adjs, diffvec_tie_adjustment=True)
                    if submethod not in predictions:
                        predictions[submethod] = dict()
                    if submethod + "-01ties" not in predictions:
                        predictions[submethod + "-01ties"] = dict()
                    predictions[submethod][idi] = prediction
                    predictions[submethod + "-01ties"][idi] = prediction_ties


    ##### EVALUATION

    # all-yes baseline
    ks = sorted(list(predictions['gold'].keys()))
    y = [predictions['gold'][idi] for idi in ks]
    method = "allYES"
    yhat = ['yes' for k in ks]
    p, r, f = prf_macro(y, yhat)
    accuracy = acc(y, yhat)
    print(method + ',' + ",".join(['%0.3f' % v for v in [accuracy, p, r, f]]))


    for method in predictions:
        if method == "gold":
            continue
        yhat = [predictions[method][idi] for idi in ks]
        p, r, f = prf_macro(y, yhat)
        accuracy = acc(y, yhat)
        print(method + ',' + ",".join(['%0.3f' % v for v in [accuracy, p, r, f]]))


