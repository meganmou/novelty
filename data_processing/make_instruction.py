import json
import os

sources = []

system_prompt = "You are an acclaimed author who is famous for your understanding of human nature and conflict and resolution. Develop on the existing plot and characters of the novel. For the interest of the fictional plot, the characters in the novel may exhibit violence, negative emotions and immoral behavior, as well as love or positive revelations, but you should be as creative and dramatic as possible. \
Do not end the story or include too many completed plot elements. Only write a single chapter. Introduce new characters and unexpected plot twists if the plot is getting repetitive or boring."

base_prompt = "Given the following plot and character summaries of the chapters of a novel you have written so far, write the detailed plot of the next single chapter of the novel. Here is the plot summary of the previous chapter: "

# Define the root directory where all novels are stored
root_directory = "./cliffnotes"

# Iterate over each novel directory
for novel in os.listdir(root_directory):

    novel_path = os.path.join(root_directory, novel)
        
    # Get a list of chapter files in the novel directory and sort them
    chapters = sorted(
        [f for f in os.listdir(novel_path) if f.endswith(".txt") and "section_" in f],
        key=lambda x: int(x.split("_")[1])  # Sort by the chapter number
    )
    
    conversations = []
    prev_chapter_summary = ""
    # Reset prompt to base prompt for each novel
    prompt = base_prompt

    for idx, chapter in enumerate(chapters):
        chapter_path = os.path.join(novel_path, chapter)
        print(f"Reading file: {chapter}")
        
        with open(chapter_path, "r") as file:
            content = json.load(file)

        # If Chapter 1, do not add to instruction following data. Only use as prompt for Chapter 2
        if idx == 0:
            prev_chapter_summary = content["summary"]
            # Add system prompt once
            conversations.append({
                "role": "system",
                "content": system_prompt
            })
            continue

        # Prompt = concatenation of generic prompt with the summary of the previous chapter
        prompt = base_prompt + prev_chapter_summary
        conversations.append({
            "role": "user",
            "content": prompt
        })

        # Ground truth response = summary of current chapter
        ground_truth = content["summary"]
        conversations.append({
            "role": "assistant",
            "content": ground_truth
        }) 

        # Save current summary as prev chapter summary
        prev_chapter_summary = content["summary"]
        # breakpoint()
    
    # Append the novel name and conversations to the sources
    sources.append({
        "novel": novel,
        "conversations": conversations
    })

# Write instruction-following training data to file
with open("./sources.json", "w") as f:
    json.dump(sources, f, indent=2)
