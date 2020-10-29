

'''
Read the QA file, extract the info so I can get the representations
'''

import pandas as pd
from nltk import word_tokenize
import pickle



instances = []

df = pd.read_csv("indirect-answers.combined.imdb-predictions.csv", sep='\t')

adj_classifications = ['adjectives_cnn.txt', 'adjectives_dialogact.txt']
adjective_cond1 = df['Classification'] == adj_classifications[0]
adjective_cond2 = df['Classification'] == adj_classifications[1]
kept = df[adjective_cond1 | adjective_cond2]


for i, r in kept.iterrows():
	### sentence 1 (a) is the question
	instance = dict(r)
	question = " ".join(r["Question"].strip().split()[1:]).lower()
	question = word_tokenize(question)
	adjA = r['AdjectiveA'].strip().lower()
	adjB = r['AdjectiveB'].strip().lower()

	if adjA == "all right":	
		position1 = [question.index("all"), question.index("right")]
	elif adjA == "in trouble":	
		position1 = [question.index("in"), question.index("trouble")] # because there is only one of each...
	elif "/" in adjA:
		position1 = [question.index(adjA.split("/")[0])]
	elif adjA == "enough to be leader":
		position1 = [question.index("enough")]
	else:	
		position1 = [question.index(adjA)]	

	instance['sentence_words1'] = tuple(question)
	instance['position1'] = tuple(position1)

	answer = " ".join(r["Answer"].strip().split()[1:]).lower()
	answer = word_tokenize(answer)

	if r['Negation']  == "UN- prefix":
		position2 = [answer.index('un' + adjB)]
	elif r['Negation'] == 'IN- prefix':
		position2 = [answer.index('in' + adjB)]
	elif "/" in adjB:
		position2 = [answer.index(adjB.split("/")[0])]
	elif len(adjB.split()) > 1:
		position2 = []
		for w in adjB.split():
			position2.append(answer.index(w))
	else:
		position2 = [answer.index(adjB)]
	instance['sentence_words2'] = tuple(answer)
	instance['position2'] = tuple(position2)

	instances.append(instance)



pickle.dump(instances, open("QA_instances.pkl", "wb"))
