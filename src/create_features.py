#/usr/bin/env python3

import sys
sys.path.append("src")
from utils import save_keys_pickle, DEVICE
import torch, string
from transformers import logging, BertTokenizer, BertForMaskedLM

logging.set_verbosity_error()

def load_model():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-cased').to(DEVICE).eval()
    return bert_tokenizer,bert_model

def encode(tokenizer, text_sentence):
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(text_sentence)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def decode(tokenizer, pred_idx):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return tokens

def get_all_predictions(text_sentence, top_k=8):
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    input_ids = input_ids.to(DEVICE)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    predict = predict[0, mask_idx, :]

    softmax = torch.nn.functional.softmax(predict, dim=0)
    top_softmax = softmax.topk(k=4, dim=0).values
    top_logit = predict.topk(k=4, dim=0).values
    top_features = torch.cat((top_softmax, top_logit))
    # set topk to its triple because some tokens may get condensed together
    bert = decode(bert_tokenizer, predict.topk(2*top_k).indices.tolist()[:top_k])
    return bert, top_features

HISTORY_LEN = 8
if __name__ == "__main__":
    bert_tokenizer, bert_model = load_model() 
    with open("data/brown.txt", "r") as f:
        text = f.read().replace("\n", "")
        text = text[:100000]
        tokenized = bert_tokenizer.tokenize(text)
    hits = 0
    i = HISTORY_LEN
    total_count = 0
    data = []
    while i < len(tokenized):
        total_count += 1
        predict_word = tokenized[i]
        probe_i = i+1
        while probe_i != len(tokenized) and tokenized[probe_i].startswith("##"):
            predict_word += tokenized[probe_i].lstrip("##")
            probe_i += 1
        context = tokenized[i-HISTORY_LEN:probe_i+HISTORY_LEN]
        context[HISTORY_LEN] = bert_tokenizer.mask_token
        res, features = get_all_predictions(context)
        # print(context, predict_word in res, res)
        hits += predict_word in res
        i = probe_i
        data.append((features.cpu().numpy(),  predict_word in res))
    print(hits,'/',total_count)
    save_keys_pickle(data, "data/brown_c1.pkl")