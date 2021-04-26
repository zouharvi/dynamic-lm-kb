#!/usr/bin/env python3

import sys
sys.path.append("src")
from utils import read_keys_pickle, DEVICE
import argparse
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, model, batchSize=256, learningRate=0.0001):
        super().__init__()

        if model == 1:
            self.layers = [
                nn.Linear(8, 1),
                nn.Sigmoid(),
            ]
        else:
            raise Exception("Unknown model specified")
        self.all_layers = nn.Sequential(*self.layers)

        self.model = model
        self.batchSize = batchSize
        self.learningRate = learningRate

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learningRate
        )
        self.criterion = nn.BCELoss()
        self.to(DEVICE)

    def forward(self, x):
        return self.all_layers(x)

    def trainModel(self, data, epochs):
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=data, batch_size=self.batchSize, shuffle=True
        )
        data_x = torch.Tensor([x[0] for x in data]).to(DEVICE)
        data_y = torch.Tensor([x[1] for x in data]).type(torch.BoolTensor).to(DEVICE)
        total = data_x.shape[0]
        print(f"LM accuracy (top k): {data_y.sum()/total*100:.2f}%")

        for epoch in range(epochs):
            self.train(True)
            for sample, gold_pred in self.dataLoader:
                sample = sample.to(DEVICE)
                gold_pred = (gold_pred*1.0).to(DEVICE)
                # Predictions
                output = self(sample).reshape(-1)
                # Calculate Loss
                loss = self.criterion(output, gold_pred)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.train(False)
                THRESHOLD = 0.5
                output = self(data_x).reshape(-1).gt(THRESHOLD)
                TP_TN = (output == data_y).sum()
                TP = (output).logical_and(data_y).sum()
                TN = (~output).logical_and(~data_y).sum()
                FP = (output).logical_and(~data_y).sum()
                FN = (~output).logical_and(data_y).sum()
                print(
                    f"BCE: {loss.data:.4f}, acc: {TP_TN/total*100:.2f}%,",
                    f"TP: {TP/total*100:.2f}%, FP: {FP/total*100:.2f}%, TN: {TN/total*100:.2f}%, FN: {FN/total*100:.2f}%"
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys-in', default="data/brown_c1.pkl")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', default=1000, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = read_keys_pickle(args.keys_in)
    model = Classifier(model=1)
    print(model)
    model.trainModel(data, args.epochs)
