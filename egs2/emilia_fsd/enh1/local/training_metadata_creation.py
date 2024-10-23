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
parser.add_argument('--train_metadata_path', type=str, default='/ocean/projects/cis210027p/ajoshi5/Datasets/emilia/metadata/train')
parser.add_argument('--num_shards', type=int, default=101)
parser.add_argument('--num_spk', type=int, default=100000)
parser.add_argument('--num_hours', type=int, default=100)
args = parser.parse_args()

EMILIA_ROOT = args.emilia_base_path
RANDOM_STATE=1
np.random.seed(RANDOM_STATE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# Load full data
NUM_SHARDS = args.num_shards
NUM_SPEAKERS = args.num_spk
NUM_HOURS = args.num_hours
PATH = os.path.join("full_data")

shard_list = []
for i in range(NUM_SHARDS):
    shard_list.append(pd.read_csv(os.path.join(PATH, f"shard_{i}.csv")))

full_meta_data = pd.concat(shard_list, ignore_index=True)
full_meta_data.head()

# Commented out IPython magic to ensure Python compatibility.
# Load speaker data
speaker_df_orig = pd.read_csv(f"speaker_data_{NUM_SHARDS}shards.csv")[:NUM_SPEAKERS]

speaker_df = speaker_df_orig.sort_values(by="speaker", key=lambda x: x.map(lambda x: eval(x[-1])), ignore_index=True)
speaker_df.head()

def load_and_infer(root_path, filepath, speech2spk_embed):
    try:
      waveform, sample_rate = librosa.load(os.path.join(root_path,filepath), sr=16000)
      print(filepath)
      # waveform = resampler(waveform)
      return speech2spk_embed(torch.Tensor(waveform).to(device))
    except:
      return None

def generate_speaker_embedding(complete_data_df, speaker_list, speaker_list_orig=speaker_df_orig):
    ## To avoid I/O on multiple directories (time expensive), we access it one at a time

    data_df = pd.merge(speaker_list["speaker"], complete_data_df, on="speaker", how="left")
    df_spk = data_df.groupby("speaker", sort=False).last()
    df_spk["embedding"] = df_spk["wav"].apply(lambda x: load_and_infer(EMILIA_ROOT, x))

    ## Sort it again according to total duration
    df_spk = pd.merge(speaker_list_orig["speaker"], df_spk, on="speaker", how="left")
    df_spk.dropna(inplace=True)

    embeddings = torch.concat(df_spk["embedding"].tolist()).cpu()
    torch.save(embeddings, f"speaker_embedding_{NUM_SHARDS}_shards_{len(speaker_list)}_speakers.pt")
    return embeddings

def generate_speaker_embedding_cache(path_to_emb):
    return torch.load(path_to_emb)

def get_tsne_points(speaker_embeddings, num_dim=2):
    num_emb = speaker_embeddings.shape[0]
    perp = min(100.0, num_emb-1)
    tsne = TSNE(n_components=num_dim, random_state=RANDOM_STATE, perplexity=perp, n_jobs=-1)
    points = tsne.fit_transform(speaker_embeddings)
    np.save(f"points_{NUM_SHARDS}_shards_{num_emb}_speakers.npy", points)
    return points

def get_tsne_points_cache(path_to_points):
    return np.load(path_to_points)

def generate_tsne_plots(points, subsets, title, filename, avg_dnsmos=None, alphas=None):
  null_str = "-"
  num_subsets = len(subsets)
  num_spk = points.shape[0]
  fig, axes = plt.subplots(num_subsets, 1, figsize=(12, 12))
  axes = [axes]

  # Plotting
  for i, ax in enumerate(axes):
      ax.scatter(points[:, 0], points[:, 1], c='blue', s=10, alpha=0.075, label=f'{num_spk} speakers')
      ax.scatter(points[subsets[i], 0], points[subsets[i], 1], c='red', s=20, alpha=0.5, label=f'{len(subsets[i])} speakers')
      ax.set_title(f"Speakers: {len(subsets[i])}, DNSMOS: {round(avg_dnsmos[i],4) if avg_dnsmos is not None else null_str}, Alpha: {alphas[i] if alphas is not None else null_str}", fontsize=10)
      ax.legend()

  # plt.tight_layout()
  fig.suptitle(f"t-SNE Visualization of Speaker Embeddings: {title}", fontsize=16)
  plt.savefig(f"tsne_plot_{NUM_SHARDS}_shards_{num_spk}_speakers_{filename}.png")
  plt.show()

def generate_tsne_plots_3d(points, subsets, title, filename, avg_dnsmos=None, alphas=None):
    null_str = "-"
    num_subsets = len(subsets)
    num_spk = points.shape[0]
    fig, axes = plt.subplots(num_subsets, 1, figsize=(12, 12), subplot_kw={"projection":"3d"})
    axes = [axes]

    # Plotting
    for i, ax in enumerate(axes):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.0075, label=f'{num_spk} speakers')
        ax.scatter(points[subsets[i], 0], points[subsets[i], 1], points[subsets[i], 2], c='red', alpha=0.5, label=f'{len(subsets[i])} speakers')
        ax.set_title(f"Speakers: {len(subsets[i])}, DNSMOS: {round(avg_dnsmos[i],4) if avg_dnsmos is not None else null_str}, Alpha: {alphas[i] if alphas is not None else null_str}", fontsize=10)
        ax.legend()

    # plt.tight_layout()
    fig.suptitle(f"t-SNE Visualization of Speaker Embeddings: {title}", fontsize=16)
    plt.savefig(f"tsne_plot_{NUM_SHARDS}_shards_{num_spk}_speakers_{filename}.png")
    plt.show()

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

# Commented out IPython magic to ensure Python compatibility.
# spk_emb = generate_speaker_embedding(full_meta_data, speaker_df)
# tsne_points = get_tsne_points(spk_emb)
spk_emb = generate_speaker_embedding_cache(f"speaker_embedding_{NUM_SHARDS}_shards_{100999}_speakers.pt")[:NUM_SPEAKERS,:]
tsne_points = get_tsne_points_cache(f"points_{NUM_SHARDS}_shards_{100999}_speakers.npy")[:NUM_SPEAKERS,:]
tsne_points_3d = get_tsne_points_cache(f"points_{NUM_SHARDS}_shards_{100999}_speakers_3d.npy")[:NUM_SPEAKERS,:]
# tsne_points = get_tsne_points(spk_emb, num_dim=2)

print("Speaker embedding shape: ", spk_emb.shape)
print("TSNE shape: ", tsne_points.shape)

subset_sizes = [2000]

# Save all subsequent data to train folder
os.chdir(args.train_metadata_path)

def get_average_dnsmos(subset_indices, speaker_df=speaker_df):
  return speaker_df["dnsmos"][subset_indices].mean()

def normalize(array):
    arr_min = array.min()
    arr_max = array.max()
    return (array - arr_min)/(arr_max - arr_min)

"""## Top speakers

"""

subsets = [np.arange(i) for i in subset_sizes]
avg_dnsmos = [get_average_dnsmos(subset_ind) for subset_ind in subsets]

generate_tsne_plots(tsne_points, subsets, "Max duration", "max_duration", avg_dnsmos)
print(f"DNSMOS: {avg_dnsmos[0]}")

spk_selected = speaker_df_orig.iloc[subsets[0]]
total_time = spk_selected["duration"].sum()
print(f"Total time of speakers: {total_time/3600} hours")
train_data = create_dataset_efficient(full_meta_data, spk_selected, NUM_HOURS)
spk_selected.to_csv(f"speaker_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_top_speakers.csv", index=False)
train_data.to_csv(f"train_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_{NUM_HOURS}_hours_top_speakers.csv", index=False)
train_data.head()

total_size = train_data["duration"].sum()/3600
print(f"Total size: {total_size} hours")

num_speakers = train_data["speaker"].nunique()
print(f"Total number of speakers: {num_speakers}")

generate_tsne_plots_3d(tsne_points_3d, subsets, "Max duration", "max_duration_3d", avg_dnsmos)
del spk_selected, train_data, subsets, avg_dnsmos
gc.collect()

"""## Random Speakers"""

subsets = [np.random.choice(spk_emb.shape[0], size, replace=False) for size in subset_sizes]
avg_dnsmos = [get_average_dnsmos(subset_ind) for subset_ind in subsets]

generate_tsne_plots(tsne_points, subsets, "Randomly selected", "random", avg_dnsmos)
print(f"DNSMOS: {avg_dnsmos[0]}")

spk_selected = speaker_df_orig.iloc[subsets[0]]
total_time = spk_selected["duration"].sum()
print(f"Total time of speakers: {total_time/3600} hours")
train_data = create_dataset_efficient(full_meta_data, spk_selected, NUM_HOURS)
spk_selected.to_csv(f"speaker_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_random.csv", index=False)
train_data.to_csv(f"train_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_{NUM_HOURS}_hours_random.csv", index=False)
train_data.head()

total_size = train_data["duration"].sum()/3600
print(f"Total size: {total_size} hours")

num_speakers = train_data["speaker"].nunique()
print(f"Total number of speakers: {num_speakers}")

generate_tsne_plots_3d(tsne_points_3d, subsets, "Randomly selected", "random_3d", avg_dnsmos)
del spk_selected, train_data, subsets, avg_dnsmos
gc.collect()

"""## Diversity: Sum of Euclidean distances from all points"""

def select_subset_all(embeddings, num_select, alpha=1, speaker_df=speaker_df):

    # print(f"*** NUM_SELECT = {num_select}")
    num_embeddings = embeddings.shape[0]
    dnsmos_scores = normalize(speaker_df["dnsmos"].to_numpy())
    subset_indices = np.ones(num_select, dtype="int") * -1
    final_scores = np.zeros(num_select, dtype="float")

    # first_index = np.argmax(dnsmos_scores)
    first_index = np.random.randint(num_embeddings)
    # mean_emb = torch.mean(embeddings, axis=0)
    # distances_from_mean = np.linalg.norm(embeddings - mean_emb, axis=1)
    # first_index = np.argmax(distances_from_mean)

    subset_indices[0] = first_index

    distances = np.square(np.linalg.norm(embeddings - embeddings[first_index], axis=1))
    sum_distances = distances.copy()
    scores = alpha * normalize(distances) + (1 - alpha) * dnsmos_scores

    for i in range(1, num_select):
      scores[subset_indices] = -1
      next_index = np.argmax(scores)
      # print(f"Sample {i}: DNSMOS={dnsmos_scores[next_index]}, Distance={distances[next_index]}, Emb={spk_emb[next_index]}")
      subset_indices[i] = next_index
      final_scores[i] = scores[next_index]

      new_distances = np.square(np.linalg.norm(embeddings - embeddings[next_index], axis=1))
      sum_distances += new_distances  # Incremental update of distances
      distances = np.minimum(distances, new_distances)  # Find minimum distance to the subset
      # distances = np.maximum(distances, new_distances)  # Find maximum distance to the subset
      scores = alpha * normalize(distances) + (1 - alpha) * dnsmos_scores

    # print(tsne_points[subset_indices])
    final_scores = alpha * normalize(sum_distances[subset_indices]) + (1 - alpha) * dnsmos_scores[subset_indices]

    return subset_indices, final_scores

subsets = []
scores = []
for size in subset_sizes:
  subset, score = select_subset_all(spk_emb, size, alpha=1)
  subsets.append(subset)
  scores.append(score)
  print(f"Size {size} done!")

avg_dnsmos = [get_average_dnsmos(subset_ind) for subset_ind in subsets]

generate_tsne_plots(tsne_points, subsets, "Diversity: Euclidean Distance from all speakers", "euclid_dist_all", avg_dnsmos, alphas=len(subsets)*[1])
print(f"DNSMOS: {avg_dnsmos[0]}")

spk_selected = speaker_df_orig.iloc[subsets[0]].copy()
spk_selected["score"] = scores[0]
total_time = spk_selected["duration"].sum()
print(f"Total time of speakers: {total_time/3600} hours")
train_data = create_dataset_efficient(full_meta_data, spk_selected, NUM_HOURS)
spk_selected.to_csv(f"speaker_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_diversity.csv", index=False)
train_data.to_csv(f"train_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_{NUM_HOURS}_hours_diversity.csv", index=False)
train_data.head()

total_size = train_data["duration"].sum()/3600
print(f"Total size: {total_size} hours")

num_speakers = train_data["speaker"].nunique()
print(f"Total number of speakers: {num_speakers}")

generate_tsne_plots_3d(tsne_points_3d, subsets, "Diversity: Euclidean Distance from all speakers", "euclid_dist_all_3d", avg_dnsmos, alphas=len(subsets)*[1])
del spk_selected, train_data, subsets, avg_dnsmos
gc.collect()

alpha=0.5
subsets = []
scores = []
for size in subset_sizes:
  subset, score = select_subset_all(spk_emb, size, alpha=alpha)
  subsets.append(subset)
  scores.append(score)
  print(f"Size {size} done!")

avg_dnsmos = [get_average_dnsmos(subset_ind) for subset_ind in subsets]

generate_tsne_plots(tsne_points, subsets, "Diversity: Euclidean Distance from all speakers", "euclid_dist_all", avg_dnsmos, alphas=len(subsets)*[0.5])
print(f"DNSMOS: {avg_dnsmos[0]}")

spk_selected = speaker_df_orig.iloc[subsets[0]].copy()
spk_selected["score"] = scores[0]
total_time = spk_selected["duration"].sum()
print(f"Total time of speakers: {total_time/3600} hours")
train_data = create_dataset_efficient(full_meta_data, spk_selected, NUM_HOURS)
spk_selected.to_csv(f"speaker_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_diversity_alpha_{alpha}.csv", index=False)
train_data.to_csv(f"train_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_{NUM_HOURS}_hours_diversity_alpha_{alpha}.csv", index=False)
train_data.head()

total_size = train_data["duration"].sum()/3600
print(f"Total size: {total_size} hours")

num_speakers = train_data["speaker"].nunique()
print(f"Total number of speakers: {num_speakers}")

generate_tsne_plots_3d(tsne_points_3d, subsets, "Diversity: Euclidean Distance from all speakers", "euclid_dist_all_3d", avg_dnsmos, alphas=len(subsets)*[0.5])
del spk_selected, train_data, subsets, avg_dnsmos
gc.collect()

alpha=0.0
subsets = []
scores = []
for size in subset_sizes:
  subset, score = select_subset_all(spk_emb, size, alpha=alpha)
  subsets.append(subset)
  scores.append(score)
  print(f"Size {size} done!")

avg_dnsmos = [get_average_dnsmos(subset_ind) for subset_ind in subsets]

generate_tsne_plots(tsne_points, subsets, "Diversity: Euclidean Distance from all speakers", "euclid_dist_all", avg_dnsmos, alphas=len(subsets)*[0.5])
print(f"DNSMOS: {avg_dnsmos[0]}")

spk_selected = speaker_df_orig.iloc[subsets[0]].copy()
total_time = spk_selected["duration"].sum()
print(f"Total time of speakers: {total_time/3600} hours")
train_data = create_dataset_efficient(full_meta_data, spk_selected, NUM_HOURS)
spk_selected.to_csv(f"speaker_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_diversity_alpha_{alpha}.csv", index=False)
train_data.to_csv(f"train_data_{NUM_SHARDS}_shards_{subset_sizes[0]}_speakers_{NUM_HOURS}_hours_diversity_alpha_{alpha}.csv", index=False)
train_data.head()

total_size = train_data["duration"].sum()/3600
print(f"Total size: {total_size} hours")

num_speakers = train_data["speaker"].nunique()
print(f"Total number of speakers: {num_speakers}")

generate_tsne_plots_3d(tsne_points_3d, subsets, "Diversity: Euclidean Distance from all speakers", "euclid_dist_all_3d", avg_dnsmos, alphas=len(subsets)*[0.0])
del spk_selected, train_data, subsets, avg_dnsmos
gc.collect()



