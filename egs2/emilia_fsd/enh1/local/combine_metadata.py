# -*- coding: utf-8 -*-

# Author: Atharva Anand Joshi (atharvaa@andrew.cmu.edu)

import os
import json
# import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser(description="A script to combine json metadata files into shard-wise csv files")
parser.add_argument('--emilia_base_path', type=str, default='/ocean/projects/cis210027p/ajoshi5/Datasets/emilia')
parser.add_argument('--start_shard', type=int, default=0)
parser.add_argument('--end_shard', type=int, default=1001)
parser.add_argument('--languages', type=str, default='EN')

args = parser.parse_args()

def json_to_csv(emilia_base_path, lang, shard_start, shard_end, output_path):
    # List to hold data from all JSON files
    # combined_data = []
    lang_folder = os.path.join(emilia_base_path, lang)

    
    for shard in range(shard_start, shard_end):

        output_csv_path = os.path.join(output_path, f"shard_{shard}.csv")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = None
            header_written = False
            # Loop through all files in the folder
            
            num_digits = len(str(shard))
            zeros_to_add = (6-num_digits) * "0"
            shard_folder_name = f"{lang}-B{zeros_to_add}{shard}"
            json_folder_path = os.path.join(lang_folder, shard_folder_name)
            for file_name in os.listdir(json_folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(json_folder_path, file_name)
                    
                    # Open and read the JSON file
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        mp3_file_name = os.path.splitext(file_name)[0] + ".mp3"
                        data["wav"] = os.path.join(lang, shard_folder_name, mp3_file_name)
                        if not header_written:
                            csv_writer = csv.DictWriter(csv_file, fieldnames=data.keys())
                            csv_writer.writeheader()
                            header_written = True
                        # combined_data.append(data)
                        csv_writer.writerow(data)
                        print(f"Written {file_name}")

        print(f"*** Shard {shard} done ***")
                    
                    

    # Create a DataFrame from the list of dictionaries
    # df = pd.DataFrame(combined_data)

    # # Export DataFrame to CSV
    # df.to_csv(output_csv_path, index=False)
    # print(f"Combined data written to {output_csv_path}")

emilia_base_path = args.emilia_base_path
languages = args.languages.split(' ')
start_shard = args.start_shard
end_shard = args.end_shard
output_path = os.path.join(emilia_base_path, 'metadata','full_data')

for lang in languages:
    json_to_csv(emilia_base_path, lang, start_shard, end_shard, output_path)
