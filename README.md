# 🌴 scalar_adjs

Code for the paper:

Aina Garí Soler and Marianna Apidianaki (2020). [BERT Knows Punta Cana is not just beautiful, it's gorgeous: Ranking Scalar Adjectives with Contextualised Representations](https://arxiv.org/abs/2010.02686). In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Nov 16-20.


#### Data

The `data` folder contains:

+ The scales in the Demelo, Crowd and Wilkinson datasets. They were obtained from [this repository](https://github.com/acocos/scalar-adj/).

+ `\*\_selected_scalar_sentences.pkl` files are pickled Python objects containing the 10 sentences used for each adjective scale. Each file corresponds to a different sentence pool (ukwac, ukwac-random, flickr).

+ `coinco_data_500.pkl` is the dataset used for the evaluation described in the Supplementary Material. It contains 500 sentence pairs extracted from the CoInCo corpus, where lexical substitutes are proposed for all content words.
The sentence pairs share a word and a set of possible substitutes along with the number of annotators who proposed the substitute. The variance ratio of these frequencies is taken to be an indication of whether all substitutes fit in the sentence equally well or not. If it is high, one substitute might be much more appropriate than the others. The task was to determine, for each sentence pair, which sentence is a better fit for ALL substitutes in the set.
For the code used for this evaluation, please contact us.

#### Code

##### Extracting representations
Contextualized representations for the sentences can be generated with:

`python extract_representations --sentences ukwac`

(alternatively: ukwac-random, flickr). They will be stored in `data/`.

##### Making ranking predictions

Once representations are extracted, predictions can be generated with:

`python predict.py`

By default they will be written to a new `predictions/` directory, using ukwac sentences, and all methods of the paper will be run (baselines, diffvec-1 (+)...).
See the options available in the script for more details, and the specifities of FREQ and static methods below:

###### FREQ predictions

To generate the frequency baseline (FREQ) predictions, you should have a file with frequencies and indicate its location with --freq_file. 
It should be a zipped text file with this format:
`word[TAB]frequency\n`

###### Static word embedding predictions

To generate predictions made by static embeddings, the file "GoogleNews-vectors-negative300.magnitude" (which can be found [here](https://github.com/plasticityai/magnitude)) should be in the `data/` directory. You will also need to install the magnitude library (see the link).


##### Evaluating adjectives ranking

`python eval.py`


##### Making indirect QA predictions & evaluating

Predictions on the QA task can be generated with:

`python qa_predict.py`

This uses the file `QA_instances_representations.pkl`, which already contains the `bert-base-uncased` representations needed for this task. 


##### Obtaining sentences from ukwac

`extract_ukwac_scalar.py` was used to find sentences in the ukwac corpus containing the scalar adjectives. To use it you need to download the ukwac corpus and specify its location with the flag `--corpus_dir`. The code for filtering out sentences containing Hearst patterns is not included in this repository.









For any questions or requests feel free to contact me: aina dot gari at limsi dot fr
