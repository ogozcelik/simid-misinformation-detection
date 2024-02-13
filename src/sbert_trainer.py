# Author: Oguzhan Ozcelik
# Date: 17.01.2024
# Subject: Run script to fine-tune SBERT model via Contrastive Learning 

import argparse
import os
import random
import pandas as pd

from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # using single GPU cuda:0 

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Specify the dataset name", type=str, choices=["twitter15", "twitter16", "mide22"])
parser.add_argument("dataset_folder", help="The location of dataset folder containing train.csv and val.csv", type=str)
parser.add_argument("save_path", help="The location where to save fine-tuned SBERT model", type=str)
parser.add_argument("pair_no", help="The number of pairs used to create dataset and fine-tune SBERT model. Refer to the paper for detailed explaination.", type=int, default=5, choices=[i for i in range(1,6)])
args = parser.parse_args()
Path(os.path.join(args.save_path)).mkdir(parents=True, exist_ok=True)
print(f"DATASET: {args.dataset}, PAIR NO: {args.pair_no}")


def create_sbert_dataset(data_df: pd.DataFrame, pair_no: int):
    data_list = []
    temp_df_true = data_df[data_df['labels'] == 1].values.tolist()
    temp_df_false = data_df[data_df['labels'] == 0].values.tolist()
    for each in temp_df_true:
        tweet_id1 = each[0]
        text_1 = each[-1]

        pairs = random.sample(range(len(temp_df_true)), pair_no)
        for pair_no in pairs:
            tweet_id2 = temp_df_true[pair_no][0]
            text_2 = temp_df_true[pair_no][-1]
            data_list.append([tweet_id1, tweet_id2, text_1, text_2, 1])

        pairs = random.sample(range(len(temp_df_false)), pair_no)
        for pair_no in pairs:
            tweet_id3 = temp_df_false[pair_no][0]
            text_3 = temp_df_false[pair_no][-1]
            data_list.append([tweet_id1, tweet_id3, text_1, text_3, 0])

    for each in temp_df_false:
        tweet_id1 = each[0]
        text_1 = each[-1]

        pairs = random.sample(range(len(temp_df_true)), pair_no)
        for pair_no in pairs:
            tweet_id2 = temp_df_true[pair_no][0]
            text_2 = temp_df_true[pair_no][-1]
            data_list.append([tweet_id1, tweet_id2, text_1, text_2, 0])

        pairs = random.sample(range(len(temp_df_false)), pair_no)
        for pair_no in pairs:
            tweet_id3 = temp_df_false[pair_no][0]
            text_3 = temp_df_false[pair_no][-1]
            data_list.append([tweet_id1, tweet_id3, text_1, text_3, 1])

    return pd.DataFrame(data_list, columns=['tweet_id_1', 'tweet_id_2', 'tweet_1', 'tweet_2', 'pseudo_label'])


train_df = pd.read_csv(os.path.join(args.dataset_folder, 'train.csv'), dtype={'tweet_id': str, 'user_id': str, 'label': int})
val_df = pd.read_csv(os.path.join(args.dataset_folder, 'val.csv'), dtype={'tweet_id': str, 'user_id': str, 'label': int})

train_df.rename(columns={'label': 'labels'}, inplace=True)
val_df.rename(columns={'label': 'labels'}, inplace=True)

train_df_sbert = create_sbert_dataset(train_df, args.pair_no)
val_df_sbert = create_sbert_dataset(val_df, args.pair_no)
print('SBERT datasets are created with pseudo-labels')

temp_path = os.path.join(args.dataset_folder, "sbert_dataset_pair_"+str(args.pair_no))
Path(temp_path).mkdir(parents=True, exist_ok=True)
train_df_sbert.to_csv(os.path.join(temp_path, "train.csv"), index=False, encoding='utf8')
val_df_sbert.to_csv(os.path.join(temp_path, "val.csv"), index=False, encoding='utf8')
print(f'SBERT datasets are saved to {temp_path}')

word_embedding_model = models.Transformer("all-mpnet-base-v2", max_seq_length=128) #
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
print(f'SBERT model is compiled')


train_examples = []
for each in train_df_sbert.values:
    train_examples.append(InputExample(texts=[each[2], each[3]], label=each[4]))
val_examples = []
for each in val_df_sbert.values:
    val_examples.append(InputExample(texts=[each[2], each[3]], label=each[4]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(val_examples, batch_size=16)
train_loss = losses.ContrastiveLoss(model)

print('Fine-tuning starts...')

model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, epochs=10, warmup_steps=100, save_best_model=True, evaluation_steps=100, output_path=args.save_path)