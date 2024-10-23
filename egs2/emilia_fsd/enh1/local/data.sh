#!/usr/bin/env bash

# Author: Atharva Anand Joshi (atharvaa@andrew.cmu.edu)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# help_message=$(cat << EOF
# Usage: $0 [--parallel <true/false>] [--use_dereverb <true/false>]
#   optional argument:
#       [--parallel]: true (Default), false
#           whether to use parallel execution of the script
#       [--use_dereverb]: false (Default), true
#           whether to use dereverb or reverb references during training
# EOF
# )

# . ./db.sh
# . ./path.sh


# output_path=$PWD/data/mixtures
# output_scp=$PWD/data
output_path=/ocean/projects/cis210027p/ajoshi5/Datasets/emilia-fsd-mixtures
output_scp=$PWD/data
spk1_dir="s1"

mkdir -p "${output_scp}"

# . utils/parse_options.sh

# if [ $# -ne 0 ]; then
#     echo "${help_message}"
#     exit 1;
# fi

# if [ ! -e "${EMILIA}" ]; then
#     log "Fill the value of 'EMILIA' of db.sh"
#     exit 1
# fi
# if [ ! -e "${FSD50K}" ]; then
#     log "Fill the value of 'FSD50K' of db.sh"
#     exit 1
# fi



# ### Combine metadata shardwise
# # This step takes ~30 min per shard. Hence only use as many as required.
# echo "Combining metadata shard wise"
# python combine_metadata.py \
#   --emilia_base_path "${EMILIA}" \
#   --start_shard 0 \
#   --end_shard 101 \
#   --languages "EN"

# ### Create training metadata
# echo "Creating training metadata"
# python training_metadata_creation.py \
#   --emilia_base_path "${EMILIA}" \
#   --train_metadata_path "${EMILIA}/metadata/train" \
#   --num_shards 101 \
#   --num_spk 100000 \
#   --num_hours 100

# ### Create validation metadata
# echo "Creating validation metadata"
# python val_metadata_creation.py \
#   --emilia_base_path "${EMILIA}" \
#   --train_metadata_path "${EMILIA}/metadata/val" \
#   --shard_num 301 \
#   --num_spk 1000 \
#   --num_hours 25

# ### Create the mixtures
# for speech in "${EMILIA}/metadata/train"/*.csv; do
#   for noise in "${FSD50K}/FSD50K.ground_truth"/*.csv; do
#     speech_name=$(basename "${speech}" .csv)
#     noise_name=$(basename "${noise}" .csv)
#     python create_mixture.py \
#       --emilia_metadata_path "${speech}" \
#       --emilia_base_path "${EMILIA}" \
#        --fsd_metadata_path "${noise}" \
#        --fsd_base_path "${FSD50K}" \
#        --name_of_dataset "${speech_name}_${noise_name}" \
#        --output_path ""
#   done
# done


### create .scp files for reference audio, noise, noisy audio
echo "Generating .scp files"
for target in "${output_path}"/train*; do
  if [ -d "$target" ]; then
    # Print the directory name
    x="$(basename "${target}")"
    echo "${x}"
    mkdir -p "${output_scp}/${x}"

    ls -1 "${output_path}/${x}/wav" | \
    awk  '{split($1, lst, ".wav"); print(lst[1])}' | \
    awk -v dir="${output_path}/${x}/wav" '{printf("%s %s/%s.wav\n", $1, dir, $1)}' | sort > "${output_scp}/${x}"/wav.scp
    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]"_"lst[3]; print($1, spk)}' "${output_scp}/${x}"/wav.scp | sort > "${output_scp}/${x}"/utt2spk
    # utils/utt2spk_to_spk2utt.pl "${output_scp}/${x}"/utt2spk > "${output_scp}/${x}"/spk2utt

    sed -e "s/\/wav\//\/${spk1_dir}\//g" ./data/"${x}"/wav.scp > ./data/"${x}"/spk1.scp
    sed -e 's/\/wav\//\/noise\//g' ./data/"${x}"/wav.scp > ./data/"${x}"/noise1.scp
  fi    
done

for target in "${output_path}"/val*; do
  if [ -d "$target" ]; then
    # Print the directory name
    echo "$(basename "${target}")"
    x="$(basename "${target}")"
    mkdir -p "${output_scp}/${x}"

    ls -1 "${output_path}/${x}/wav" | \
    awk  '{split($1, lst, ".wav"); print(lst[1])}' | \
    awk -v dir="${output_path}/${x}/wav" '{printf("%s %s/%s.wav\n", $1, dir, $1)}' | sort > "${output_scp}/${x}"/wav.scp
    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]"_"lst[3]; print($1, spk)}' "${output_scp}/${x}"/wav.scp | sort > "${output_scp}/${x}"/utt2spk
    # utils/utt2spk_to_spk2utt.pl "${output_scp}/${x}"/utt2spk > "${output_scp}/${x}"/spk2utt

    sed -e "s/\/wav\//\/${spk1_dir}\//g" ./data/"${x}"/wav.scp > ./data/"${x}"/spk1.scp
    sed -e 's/\/wav\//\/noise\//g' ./data/"${x}"/wav.scp > ./data/"${x}"/noise1.scp
  fi    
done
