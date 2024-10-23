# -*- coding: utf-8 -*-

# Author: Atharva Anand Joshi (atharvaa@andrew.cmu.edu)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import gc
import argparse

parser = argparse.ArgumentParser(description="A script to combine json metadata files into shard-wise csv files")
parser.add_argument('--emilia_base_path', type=str, default='/ocean/projects/cis210027p/ajoshi5/Datasets/emilia')
parser.add_argument('--val_metadata_path', type=str, default='/ocean/projects/cis210027p/ajoshi5/Datasets/emilia/metadata/val')
parser.add_argument('--shard_num', type=int, default=301)
parser.add_argument('--num_spk', type=int, default=1000)
parser.add_argument('--num_hours', type=int, default=25)
args = parser.parse_args()

EMILIA_ROOT = args.emilia_base_path
RANDOM_STATE=1
np.random.seed(RANDOM_STATE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# Load full data

NUM_SHARD = args.shard_num   # Single shard not [0-300]
NUM_SPEAKERS = args.num_spk
NUM_HOURS = args.num_hours
PATH = os.path.join("full_data")

# shard_list = []
# for i in range(NUM_SHARDS):
#     shard_list.append(pd.read_csv(os.path.join(PATH, f"shard_{i}.csv")))

full_meta_data = pd.read_csv(os.path.join(PATH, f"shard_{NUM_SHARD}.csv"))
full_meta_data.head()

total_size = full_meta_data["duration"].sum()/3600
print(f"Total size: {total_size} hours")

num_speakers = full_meta_data["speaker"].nunique()
print(f"Total number of speakers: {num_speakers}")

# Commented out IPython magic to ensure Python compatibility.
# %cd /ocean/projects/cis210027p/ajoshi5/Datasets/emilia/metadata/val
num_utterances_df = full_meta_data.loc[:,["speaker","duration","dnsmos"]]
num_utterances_df.loc[:,"num_utt"] = num_utterances_df.loc[:,"speaker"]

num_utterances_df = num_utterances_df.groupby("speaker").agg(
    {
        "num_utt": "count",
        "duration": "sum",
        "dnsmos": "mean"
    }
).sort_values("duration", ascending=False)
num_utterances_df.to_csv(f"val_speaker_data_{NUM_SHARD}shard.csv")
print(num_utterances_df.head(50))

def get_nth_or_skip(group, n):
    if len(group) > n:
        return group.iloc[n]
    else:
        return None

def create_dataset_efficient(complete_data_df, speaker_df, num_hours):

    # print(num_utt_df.head())
    speaker_list = speaker_df["speaker"]
    data_df = pd.merge(speaker_list, complete_data_df, on="speaker", how="left")

    ### Shuffle the dataset
    data_df = data_df.groupby("speaker").apply(lambda x: x.sample(frac=1, random_state=RANDOM_STATE)).reset_index(drop=True)

    ### Create output dataset
    out_df = pd.DataFrame(columns=data_df.columns)

    total_duration_data_df = data_df["duration"].sum()
    if total_duration_data_df / 3600 < num_hours:
        print("Data too small!")
        return data_df


    duration = 0
    index = 0
    hours = 0

    while(duration <= num_hours * 3600):
        # df_spk = data_df.groupby("speaker").nth(index, dropna="any")
        df_spk = data_df.groupby("speaker",group_keys=False)[data_df.columns.tolist()].apply(lambda x: get_nth_or_skip(x, index))
        df_spk["duration_cumu"] = df_spk["duration"].cumsum()
        if (duration + df_spk["duration"].sum()) < num_hours * 3600:
            out_df = pd.concat([out_df, df_spk.drop(columns=["duration_cumu"])], ignore_index=True)
        else:
            df_spk = df_spk[df_spk["duration_cumu"] <= 3600*num_hours - duration]
            out_df = pd.concat([out_df, df_spk.drop(columns=["duration_cumu"])], ignore_index=True)
            break

        duration += df_spk.duration.sum()
        # print(f"{out_df['duration'].sum()//3600} Hours Done!")
        index += 1



    return out_df.dropna().sort_values(by="speaker", key=lambda x: x.map(lambda x: eval(x[-1])), ignore_index=True).reset_index(drop=True)

# Save all subsequent data to val folder
os.chdir(args.val_metadata_path)

num_utterances_df = pd.read_csv(f"val_speaker_data_{NUM_SHARD}shard.csv")
val_data = create_dataset_efficient(full_meta_data, num_utterances_df, NUM_HOURS)
val_data.to_csv(f"val_{NUM_SHARD}_shard_{NUM_SPEAKERS}_speakers_{NUM_HOURS}_hours.csv", index=False)

