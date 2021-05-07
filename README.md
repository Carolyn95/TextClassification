# TextClassification
A pipeline for aspect-based sentiment classification, mining and summarizing opinions from text about specific entities and their aspects.
Text input -> seq2seq IOB tagging -> pre-trained embedding vectors -> LSTM

Data Engineering:
  Text cleaning, contraction expansion, lemmatization
End to end modelling and inferencing pipeline:
  Convert text sequence into embedding vectors
  Create PoS tags
  Sequence to sequence IOB2 tagging for OTE(opinion-term-expressions) identification

Example input:
"Service was terribly slow and the restaurant was noisy, but the waiter was friendly and the calamari was very delicious."
Example output:
(OTE target words, entity:aspect, polarity)
("service", SERVICE:GENERAL, negative)
("restaurant", AMBIENCE:GENERAL, negative)
("waiter", SERVICE:GENERAL, positive)
("calamari", FOOD:QUALITY, positive)

All data files found at http://alt.qcri.org/semeval2016/task5/.


