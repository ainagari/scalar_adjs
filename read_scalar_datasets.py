
import os

def read_scales(dirname):
    '''
    Read all comparable pairs from the same scale from scal term files in dirname;
    keep track of which file each pair comes from
    Returns: a dict of {filename: set([(w1,w2),...])}
    '''
    termsfiles = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    rankings = dict()
    for tf in termsfiles:
        tf_clean = os.path.basename(tf).replace('.terms','')
        ranking = []
        with open(tf, 'r') as fin:
            for line in fin:
                __, w = line.strip().split('\t')
                ranking.append(w)

        rankings[tf_clean] = ranking
    return rankings

