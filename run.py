# Author: Oguzhan Ozcelik
# Date: 17.01.2024
# SUbject: Run script of SiMiD model 

import argparse
import sys
from src.simid import SiMidTrainer
import os
import pandas as pd
import json
from math import log
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # using single GPU cuda:0 

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Specify the mode", type=str, choices=["use-community", "no-community"])
parser.add_argument("dataset_folder", help="The location of dataset folder containing train.csv and val.csv", type=str)
parser.add_argument("sbert_path", help="The location of the fine-tuned SBERT model", type=str, default="/src/contrastive_sbert/")
parser.add_argument("save_path", help="The location where to save the SVM model", type=str)

args = parser.parse_args()
Path(os.path.join(args.save_path)).mkdir(parents=True, exist_ok=True)


related_tweets = pd.read_csv(os.path.join(args.dataset_folder, 'id2tweet.csv'), dtype=str)

id2tweet = {}
for k, v in zip(related_tweets.tweet_id, related_tweets.text):
    id2tweet[k] = v


model = SiMidTrainer(train_data=pd.read_csv(os.path.join(args.dataset_folder, 'train.csv'), dtype={'tweet_id': str, 'user_id': str, 'label': int, 'text':str}),
                    val_data=pd.read_csv(os.path.join(args.dataset_folder, 'val.csv'), dtype={'tweet_id': str, 'user_id': str, 'label': int, 'text':str}),
                    test_data=pd.read_csv(os.path.join(args.dataset_folder, 'train.csv'), dtype={'tweet_id': str, 'user_id': str, 'label': int, 'text':str}),
                    community=pd.read_csv(os.path.join(args.dataset_folder, 'engagers.csv'), dtype=str),
                    id2tweet=id2tweet,
                    mode=[args.mode],
                    sbert_path=args.sbert_path,
                    save_path=args.save_path
                    )

model.train()
