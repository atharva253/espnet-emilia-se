# -*- coding: utf-8 -*-

# Author: Atharva Anand Joshi (atharvaa@andrew.cmu.edu)

import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import argparse

# Set random seed for reproducibility
RANDOM_SEED = 1
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

parser = argparse.ArgumentParser(description="A script to generate mixture files and metadata")
parser.add_argument('--emilia_metadata_path', type=str, required=True)
parser.add_argument('--emilia_base_path', type=str, required=True)
parser.add_argument('--fsd_metadata_path', type=str, required=True)
parser.add_argument('--fsd_base_path', type=str, required=True)
parser.add_argument('--fsd_split', type=str, default="FSD50K.dev_audio")
parser.add_argument('--name_of_dataset', type=str, required=True)
parser.add_argument('--output_path', type=str, default=".")
parser.add_argument('--sr', type=int, default=16000)

args = parser.parse_args()

# Load metadata for the Emilia and FSD50K datasets
emilia_metadata = pd.read_csv(args.emilia_metadata_path)
fsd50k_metadata = pd.read_csv(args.fsd_metadata_path)

# Define paths for clean speech and noise datasets
EMILIA_PATH = args.emilia_base_path
FSD_PATH = os.path.join(args.fsd_base_path, args.fsd_split)

# Output directories
output_dir = os.path.join(args.output_path, args.name_of_dataset)
s1_dir = os.path.join(output_dir, 's1')
noise_dir = os.path.join(output_dir, 'noise')
wav_dir = os.path.join(output_dir, 'wav')

# Create directories if they don't exist
os.makedirs(s1_dir, exist_ok=True)
os.makedirs(noise_dir, exist_ok=True)
os.makedirs(wav_dir, exist_ok=True)

# Parameters
snr_range = (-5, 20)  # SNR levels in dB


# Function to calculate scaling factor for the noise based on the desired SNR
def calculate_noise_scaling(clean_speech, noise, snr_db):
    clean_rms = np.sqrt(np.mean(clean_speech**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    snr_linear = 10**(snr_db / 20)  # Convert SNR from dB to linear
    scaling_factor = clean_rms / (snr_linear * noise_rms)
    return scaling_factor

# Function to mix speech and noise at a given SNR
def mix_speech_noise(clean_speech, noise, snr_db):
    # Calculate scaling factor for noise to achieve the desired SNR
    scaling_factor = calculate_noise_scaling(clean_speech, noise, snr_db)
    scaled_noise = noise * scaling_factor
    noisy_speech = clean_speech + scaled_noise
    return noisy_speech, scaling_factor

# Function to generate noisy speech with random SNR
def generate_noisy_speech(emilia_metadata, fsd50k_metadata, snr_range, output_dir):
    noise_file_idx = 0  # Start at the first noise file
    
    # Initialize an empty list to hold metadata for noisy speech
    noisy_speech_metadata = []
    
    for idx, row in emilia_metadata.iterrows():
        # Load clean speech
        clean_speech_file = os.path.join(EMILIA_PATH, row['wav'])
        clean_speech, sr = librosa.load(clean_speech_file, sr=args.sr)

        # Ensure all noise files are used evenly
        if noise_file_idx >= len(fsd50k_metadata):
            random.shuffle(fsd50k_metadata)  # Reshuffle when all files are used
            noise_file_idx = 0

        # Select the next noise file from FSD50K metadata
        noise_row = fsd50k_metadata.iloc[noise_file_idx]
        noise_file_idx += 1

        # Load noise based on the metadata filename
        noise_file = os.path.join(FSD_PATH, str(noise_row['fname'])+'.wav')
        noise, _ = librosa.load(noise_file, sr=args.sr)

        # Trim or pad noise to match clean speech duration
        if len(noise) < len(clean_speech):
            noise = np.tile(noise, int(np.ceil(len(clean_speech) / len(noise))))[:len(clean_speech)]
        else:
            noise = noise[:len(clean_speech)]

        # Randomly select an SNR level from the range (seeded)
        snr_db = random.uniform(*snr_range)

        # Mix clean speech and noise, and get the scaling factor
        noisy_speech, scaling_factor = mix_speech_noise(clean_speech, noise, snr_db)

        # Define output file paths
        noisy_output_file = os.path.join(wav_dir, f'{row.id}_{noise_row.fname}_SNR{snr_db:.2f}.wav')
        clean_output_file = os.path.join(s1_dir, f'{row.id}_{noise_row.fname}_SNR{snr_db:.2f}.wav')
        noise_output_file = os.path.join(noise_dir, f'{row.id}_{noise_row.fname}_SNR{snr_db:.2f}.wav')

        # Save the noisy speech
        sf.write(noisy_output_file, noisy_speech, sr)

        # Save the clean speech file to the s1 folder (if it doesn't already exist)
        if not os.path.exists(clean_output_file):
            sf.write(clean_output_file, clean_speech, sr)

        # Save the noise file to the noise folder (if it doesn't already exist)
        if not os.path.exists(noise_output_file):
            sf.write(noise_output_file, noise, sr)

        # Append the metadata for this noisy speech
        noisy_speech_metadata.append({
            'speaker': row['speaker'],
            'clean_speech_path': row["wav"],
            'noise_category': noise_row['labels'],
            'noise_file_path': os.path.join(args.fsd_split, str(noise_row["fname"]) + ".wav"),
            'noisy_audio_path': noisy_output_file,
            'scaling_factor': scaling_factor
        })
        print(noisy_output_file)
    
    return noisy_speech_metadata


# Shuffle FSD50K metadata at the beginning (seeded) to ensure random noise selection
fsd50k_metadata = fsd50k_metadata.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Generate noisy speech with random SNR levels
generate_noisy_speech(emilia_metadata, fsd50k_metadata, snr_range, output_dir)

# Convert noisy speech metadata to DataFrame and save it to a CSV file
noisy_speech_metadata_df = pd.DataFrame(noisy_speech_metadata)
noisy_speech_metadata_df.to_csv(os.path.join(output_dir, 'noisy_speech_metadata.csv'), index=False)

print('Noisy speech files and metadata generated successfully.')