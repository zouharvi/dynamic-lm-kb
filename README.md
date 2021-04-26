# Limited LM KB

Coarse prototype of LM with limited KB access. Assume `data/brown.txt`.
Features are: `(top 4 logits, top 4 softmax probs)`, prediction is `(masked word in top 8 predictions)`. Currently context of 8+8 is given to the BERT model (`bert-base-cased`). The top 8 accuracy is `53.71%`. 

```
./src/create_features.py # first 100000 tokens using BERT
./src/classifier.py 
```

Classifier results (threshold `p=0.5`):

```
BCE: 0.6183, acc: 66.79%
TP: 41.01%, FP: 20.52%, TN: 25.78%, FN: 12.70%
```