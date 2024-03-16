import os
import json

def concatenate_json_files(folder_path):
    concatenated_data = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r') as file:
                data = json.load(file)
                concatenated_data.append(data)
    return concatenated_data

def write_to_json(concatenated_data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(concatenated_data, json_file, indent=4)

def count_num_chapters(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        num_chapters = 0
        for novel in data:
            num_chapters += (len(novel["conversations"]) / 2)
        print(f"Number of chapters in {file_path}: {num_chapters}")

if __name__ == "__main__":
    folder_path = "new_training_data"
    output_file = "new_concatenated_all_training_data.json"

    concatenated_data = concatenate_json_files(folder_path)
    write_to_json(concatenated_data, output_file)
    # count_num_chapters(output_file)
