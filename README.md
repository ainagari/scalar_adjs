# 🌴 BERT Knows Punta Cana is not just beautiful, it's gorgeous: Ranking Scalar Adjectives with Contextualised Representations 🌴

Code for the papers:

Aina Garí Soler and Marianna Apidianaki (2020). [BERT Knows Punta Cana is not just beautiful, it's gorgeous: Ranking Scalar Adjectives with Contextualised Representations](https://www.aclweb.org/anthology/2020.emnlp-main.598.pdf). In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Nov 16-20.

**code coming soon** Aina Garí Soler and Marianna Apidianaki (2021). [Scalar Adjective Identification and Multilingual Ranking](https://arxiv.org/abs/2105.01180). In Proceedings of the 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021), Jun 6-11.


Part of this code and data were obtained or adapted from [this repository](https://github.com/acocos/scalar-adj/). (Cocos et al., 2018).


### Data

The `data` folder contains:

+ The scales in the Demelo, Crowd and Wilkinson datasets.

+ `*\_selected_scalar_sentences.pkl` files are pickled Python objects containing the 10 sentences used for each adjective scale. Each file corresponds to a different sentence pool (ukwac, ukwac-random, flickr).


+ `QA_instances_representations.pkl` files are pickled Python objects containing `bert-base-uncased` representations for the adjectives in the indirect QA dataset. The original dataset can be found [here](https://raw.githubusercontent.com/cgpotts/iqap/master/ACL2010/indirect-answers.combined.imdb-predictions.csv).


+ `coinco_data_500.pkl` is the dataset used for the evaluation described in the Supplementary Material. It contains 500 sentence pairs extracted from the CoInCo corpus, where lexical substitutes are proposed for all content words.
The sentence pairs share a word and a set of possible substitutes along with the number of annotators who proposed the substitute. The variance ratio of these frequencies is taken to be an indication of whether all substitutes fit in the sentence equally well or not. If it is high, one substitute might be much more appropriate than the others. The task was to determine, for each sentence pair, which sentence is a better fit for ALL substitutes in the set.
For the code used for this evaluation, please contact us.

The `multilingual_scales` folder contains the MULTI-SCALE dataset, with translations of the DeMelo and Wilkinson datasets to French (fr), Spanish (es) and Greek (el). The folders follow the same structure as the English scales in the `data` folder. The file `all_translations.csv` contains the whole dataset including the original English scales.

The `scal-rel` folder contains the SCAL-REL dataset (`scal-rel_dataset.csv`). Each line corresponds to one adjective in the dataset: `ADJECTIVE [TAB] CLASS [TAB] SET`. `CLASS` can be `SCALAR` or `RELATIONAL`. `SET` can be `train`,`dev` or `test`.


### Code

#### Extracting BERT Representations
Contextualized representations for (ukwac, ukwac-random, flickr) sentences can be generated with the following command. They will be stored in `data/`. 

`python extract_representations --sentences ukwac`

#### Making Ranking Predictions

This is the command to generate predictions once representations have been extracted. By default they are made with `ukwac` sentences and are written to a new `predictions/` directory. Unless otherwise specified, the script will generate predictions for all methods in the paper (baselines & diffvecs). See the options available in the script for more details, and the specificities of the FREQ baseline and the static methods just below.

`python predict.py`

##### FREQ predictions

To generate frequency baseline (FREQ) predictions, you should have a file with frequencies and indicate its location with the flag `--freq_file`. 
It should be a zipped text file with this format:
`word[TAB]frequency\n`

##### Static word embedding predictions

To generate predictions made by static embeddings, the file `GoogleNews-vectors-negative300.magnitude` (which can be downloaded [here](https://github.com/plasticityai/magnitude)) should be in the `data/` directory. You will also need to install the magnitude library (see the link provided).

#### Evaluating Adjective Ranking

Once predictions are saved you can evaluate them with the `eval.py` script. By default results will be written to a new folder `results/`.

`python eval.py`


#### Making indirect QA predictions & evaluating

Predictions on the QA task can be generated with:

`python qa_predict.py`


#### Obtaining sentences from ukwac

`extract_ukwac_scalar.py` was used to find sentences in the ukwac corpus containing the scalar adjectives. To use it you need to download the ukwac corpus and specify its location with the flag `--corpus_dir`. The code for filtering out sentences containing Hearst patterns is not included in this repository but you can contact me


### Citation

If you use the code in this repository, please cite our paper:
```
@inproceedings{gari2020puntacana,
    author    = {Gar\'i Soler, Aina and Apidianaki, Marianna},
    title     = {{BERT Knows Punta Cana is not just \textit{beautiful}, it's \textit{gorgeous}: Ranking Scalar Adjectives with Contextualised Representations}},
    booktitle   = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year      = {2020},
    month     = {nov},
    url       = {https://arxiv.org/abs/2010.02686}
}
```


### References

Anne Cocos, Skyler Wharton, Ellie Pavlick, Marianna Apidianaki, and Chris Callison-Burch (2018). [Learning Scalar Adjective Intensity from Paraphrases](https://www.aclweb.org/anthology/D18-1202/). In Proceedings  of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1752–1762, Brussels, Belgium. Association for Computational Linguistics.

### Contact

For any questions or requests feel free to contact me: aina dot gari at limsi dot fr
