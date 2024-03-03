import json
import os
from datasets import load_dataset

def format_conversation(row):
    # Template for conversation turns in ChatML format
    template="<|im_start|>system\n{sys}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"

    turns=row["conversations"]

    conversation=[]

    system = turns[0]

    for i in range(1, len(turns), 2):
        # Assuming the conversation always alternates between user and assistant
        question = turns[i]
        answer = turns[i+1]

        conversation.append(
            template.format(
                sys = system["content"],
                q=question["content"],
                a=answer["content"],
                ))
    return {"text": "\n".join(conversation)}

with open("./data_processing/sources.json", "r") as file:
    dataset = json.load(file)

formatted_dataset = []
for entry in dataset:
    formatted_entry = format_conversation(entry)
    formatted_dataset.append(formatted_entry)

with open("formatted_sources.json", "w") as outfile:
    json.dump(formatted_dataset, outfile, indent=2)
