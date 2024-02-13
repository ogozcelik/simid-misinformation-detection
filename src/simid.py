import os.path
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import random
import pickle
from sklearn.metrics import classification_report
from keras import backend as K
from sklearn import svm
from pathlib import Path


class SiMidTrainer:
    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame,
                 community: pd.DataFrame,
                 id2tweet: dict, mode: list, sbert_path: str,
                 save_path: str):

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.community = community
        self.mode = mode
        self.save_path = save_path
        self.id2tweet = id2tweet
        self.sbert_encoder = SentenceTransformer(sbert_path)
        self.community_features = {}
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def cosine_similarity(vec1: list, vec2: list):
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def generate_community_features(self):

        for row in tqdm(self.community.values, total=len(self.community)):
            user_id = row[0]
            related_tweets_list = row[-1].split("%")
            related_tweet_texts = [self.id2tweet[tweet_id] for tweet_id in related_tweets_list]
            com_content = None

            if "use-community" in self.mode:
                CLS_batch = self.sbert_encoder.encode(related_tweet_texts)
                com_content = CLS_batch.mean(axis=0)

            self.community_features[user_id] = {'com_content': com_content}

    def check_similarity(self, tweet_id, text):
        input_object = {'content': self.sbert_encoder.encode(text)}

        community_cos_content = []
        for each in self.community_features.keys():
            if "use-community" in self.mode:
                community_cos_content.append(self.cosine_similarity(input_object['content'],
                                             self.community_features[each]['com_content']))

        return np.array(community_cos_content)

    def train(self):
        if self.mode != ['no-community']:
            self.generate_community_features()

        train_features = []
        for text, id in tqdm(zip(self.train_data.text.to_list(), self.train_data.tweet_id.to_list()),
                             total=len(self.train_data)):
            feature = None
            if self.mode == ['no-community']:
                feature = np.array(self.sbert_encoder.encode(text))
            elif self.mode == ['use-community']:
                feature = self.check_similarity(tweet_id=id, text=text)
            else:
                exit("You can either use or do not use communities. There is no other option.")

            train_features.append(feature)
        x_train = np.array(train_features)
        y_train = self.train_data.label.values
        print("Train Data is loaded.")

        val_features = []
        for text, id in tqdm(zip(self.val_data.text.to_list(), self.val_data.tweet_id.to_list()),
                             total=len(self.val_data)):
            feature = None
            if self.mode == ['no-community']:
                feature = np.array(self.sbert_encoder.encode(text))
            elif self.mode == ['use-community']:
                feature = self.check_similarity(tweet_id=id, text=text)
            else:
                exit("You can either use or do not use communities. There is no other option.")

            val_features.append(feature)

        x_val = np.array(val_features)
        y_val = self.val_data.label.values
        print("Val Data is loaded.")

        SVM = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', verbose=True)
        SVM.fit(np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))
        pickle.dump(SVM, open(os.path.join(self.save_path, 'simid_svm'), 'wb'))

        test_features = []
        for text, id in tqdm(zip(self.test_data.text.to_list(), self.test_data.tweet_id.to_list()),
                             total=len(self.test_data)):
            feature = None
            if self.mode == ['no-community']:
                feature = np.array(self.sbert_encoder.encode(text))
            elif self.mode == ['use-community']:
                feature = self.check_similarity(tweet_id=id, text=text)
            else:
                exit("You can either use or do not use communities. There is no other option.")

            test_features.append(feature)

        x_test = np.array(test_features)
        y_test = self.test_data.label.values
        print("Test Data is loaded.")

        y_pred = SVM.predict(x_test)
        report = classification_report(y_test, y_pred, digits=5)
        print(report)

        with open(os.path.join(self.save_path, "scores.json"), 'w') as fp:
            json.dump(classification_report(y_test, y_pred, digits=5, output_dict=True), fp)