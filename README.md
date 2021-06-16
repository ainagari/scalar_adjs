# 🌴 BERT Knows Punta Cana is not just beautiful, it's gorgeous: Ranking Scalar Adjectives with Contextualised Representations 🌴

Code for the papers:

Aina Garí Soler and Marianna Apidianaki (2020). [BERT Knows Punta Cana is not just beautiful, it's gorgeous: Ranking Scalar Adjectives with Contextualised Representations](https://www.aclweb.org/anthology/2020.emnlp-main.598.pdf). In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Nov 16-20.

Aina Garí Soler and Marianna Apidianaki (2021). [Scalar Adjective Identification and Multilingual Ranking](https://arxiv.org/abs/2105.01180). In Proceedings of the 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021), Jun 6-11.


Part of this code and data were obtained or adapted from [this repository](https://github.com/acocos/scalar-adj/) (Cocos et al., 2018).


### Data

The `data` folder contains:

+ The scales in the Demelo, Crowd and Wilkinson datasets.

+ `selected_scalar_sentences.pkl` files are pickled Python objects containing the 10 sentences used for each adjective scale. Each file corresponds to a different sentence pool (ukwac, ukwac-random, flickr).


+ `QA_instances_representations.pkl` files are pickled Python objects containing `bert-base-uncased` representations for the adjectives in the indirect QA dataset. The original dataset can be found [here](https://raw.githubusercontent.com/cgpotts/iqap/master/ACL2010/indirect-answers.combined.imdb-predictions.csv).


+ `coinco_data_500.pkl` is the dataset used for the evaluation described in the Supplementary Material. It contains 500 sentence pairs extracted from the CoInCo corpus, where lexical substitutes are proposed for all content words.
The sentence pairs share a word and a set of possible substitutes along with the number of annotators who proposed the substitute. The variance ratio of these frequencies is taken to be an indication of whether all substitutes fit in the sentence equally well or not. If it is high, one substitute might be much more appropriate than the others. The task was to determine, for each sentence pair, which sentence is a better fit for ALL substitutes in the set.
For the code used for this evaluation, please contact us.

The `multilingual_scales` folder contains the MULTI-SCALE dataset, with translations of the DeMelo and Wilkinson datasets to French (fr), Spanish (es) and Greek (el). The folders follow the same structure as the English scales in the `data` folder. The file `all_translations.csv` contains the whole dataset including the original English scales. The files `sentences_[LANG].pkl` contain the sentences used in our experiments for each of the languages in the dataset.

The `scal-rel` folder contains the SCAL-REL dataset (`scal-rel_dataset.csv`). Each line corresponds to one adjective in the dataset: `ADJECTIVE [TAB] CLASS [TAB] SET`. `CLASS` can be `SCALAR` or `RELATIONAL`. `SET` can be `train`,`dev` or `test`. The file `relational_sentences.pkl` contains the 10 sentences per adjective used in our experiments.



### Code

#### Extracting BERT Representations
Contextualized representations for (ukwac, ukwac-random, flickr, and oscar) sentences can be generated with the following command. They will be stored in `--data_dir`. 

`python extract_representations --data_dir data/ --sentences ukwac --language en`

You can change the language with `--language [en|es|fr|el]`, use a multilingual model with `--multilingual-uncased` or `--multilingual-cased`, and exclude the last bpe unit with `--exclude_last_bpe`.

Representations for relational adjectives can be extracted running the following command, with the optional argument `--exclude_last_bpe`.

`python scalrel_extract_representations.py`

#### Making Ranking Predictions

Once representations have been extracted, to generate predictions for the experiments in our first paper (Garí Soler & Apidianaki, 2020), you can run the following command. By default predictions are made using `ukwac` sentences and are written to a new `predictions/` directory. Unless otherwise specified, the script will generate predictions for all methods in the paper (baselines & diffvecs). See the options available in the script for more details, and the specificities of the FREQ baseline and the static methods just below.

`python predict_puntacana.py`

For the predictions in our 2021 paper, you can run the following script. The differences with the experiments above mainly consist in additioOnce representations have been extracted, to generate predictions for the experiments in our first paper (Garí Soler & Apidianaki, 2020), you can run the following command. By default predictions are made using `ukwac` sentences and are written to a new `predictions/` directory. Unless otherwise specified, the script will generate predictions for all methods in the paper (baselines & diffvecs). See the options available in the script for more details, and the specificities of the FREQ baseline and the static methods just below.

`python predict_puntacana.py`

For the predictions in our 2021 paper, you can run the following script. The differences with the experiments above mainly consist in additional languages and the use of fasttext embeddings instead of word2vec.

`python predict_multilingual.py --language fr --path_to_static [PATH TO STATIC EMBEDDINGS]`nal languages and the use of fasttext embeddings instead of word2vec.

`python predict_multilingual.py --language fr --path_to_static [PATH TO STATIC EMBEDDINGS]`

##### FREQ predictions

To generate frequency baseline (FREQ) predictions for English, you should have a file with frequencies and indicate its location with the flag `--english_freq_fn`. 
It should be a zipped text file with this format:
`word[TAB]frequency\n`

The frequencies for the other languages, calculated from the [OSCAR corpus](https://oscar-corpus.com/) (Ortiz Suárez et al, 2019), are provided in a different format.


##### Static word embedding predictions

To generate predictions made by static embeddings, you need to first download them from [this repository](https://github.com/plasticityai/magnitude). For word2vec, you need `GoogleNews-vectors-negative300.magnitude`. For Fasttext embeddings, `cc.[LANGUAGE].300.magnitude`. You will also need to install the magnitude library (see the link provided). When making predictions, you can indicate the path to where these files are with `--path_to_static`.

#### Evaluating Adjective Ranking

Once predictions are saved you can evaluate them with the `eval.py` script. You can run `python eval_puntacana.py` for the 2020 paper and `python eval_multilingual.py` By default results will be written to a new folder `results/` or `rmultilingual_results/`. See the optional arguments in the respective scripts for more details.


#### Making indirect QA predictions & evaluating

Predictions on the QA task can be generated with:

`python qa_predict.py`


#### Obtaining sentences from ukwac

`extract_ukwac_scalar.py` was used to find sentences in the ukwac corpus containing the scalar adjectives. To use it you need to download the ukwac corpus and specify its location with the flag `--corpus_dir`. The code for filtering out sentences containing Hearst patterns is not included in this repository. Feel free to contact me if you need it (see e-mail at the bottom). Note that the output of this script is not ready to be used with `extract_representations.py`. You need to select the sentences that will be used and put them in a format like that of the `data/selected_scalar_sentences.pkl` files. More information/scripts on this soon.


#### Classification on SCAL-REL

Classification on the scal-rel dataset can be performed running `python scalrel_classification.py`. Before that, BERT representations need to be extracted for all adjectives (See "Extracting BERT Representations" above). A path to static embeddings (`--path_to_static`) needs to be provided.

### Citation

If you use the code in this repository, please cite our paper:
```
@inproceedings{gari-soler-apidianaki-2020-bert,
    title = {{BERT Knows Punta Cana is not just beautiful, it's gorgeous: Ranking Scalar Adjectives with Contextualised Representations}},
    author = "Gar{\'\i} Soler, Aina  and
      Apidianaki, Marianna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.598",
    doi = "10.18653/v1/2020.emnlp-main.598",
    pages = "7371--7385",  
}

@inproceedings{gari-soler-apidianaki-2021-scalar,
    title = {{Scalar Adjective Identification and Multilingual Ranking}},
    author = "Gar{\'\i} Soler, Aina  and
      Apidianaki, Marianna",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.370",
    pages = "4653--4660",   
}
```


### References

Anne Cocos, Skyler Wharton, Ellie Pavlick, Marianna Apidianaki, and Chris Callison-Burch (2018). [Learning Scalar Adjective Intensity from Paraphrases](https://www.aclweb.org/anthology/D18-1202/). In Proceedings  of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1752–1762, Brussels, Belgium. Association for Computational Linguistics.

Pedro Javier Ortiz Suárez, Benoît Sagot, and Laurent Romary. 2019. [Asynchronous pipeline for processing huge corpora on medium to low resource infrastructures](https://hal.inria.fr/hal-02148693). In Proceedings of the 7th Workshop on the Challenges in the Management of Large Corpora (CMLC-7), Cardiff, UK. Leibniz-Institut für Deutsche Sprache.



### Contact

For any questions or requests feel free to contact me: aina dot gari at limsi dot fr
