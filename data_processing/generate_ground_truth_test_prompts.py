from datasets import load_dataset
import json

dataset = load_dataset("kmfoda/booksum")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

babbitt_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Babbitt") and row["source"] == "sparknotes")
madame_bovary_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Madame Bovary") and row["source"] == "cliffnotes")
the_red_and_the_black_rows = test_dataset.filter(lambda row: row["book_id"].startswith("The Red and the Black") and row["source"] == "cliffnotes")
sense_and_sensibility_rows = test_dataset.filter(lambda row: row["book_id"].startswith("Sense and Sensibility") and row["source"] == "cliffnotes")
tempest_rows = test_dataset.filter(lambda row: row["book_id"].startswith("The Tempest") and row["source"] == "cliffnotes")

def convert_to_train(novel_rows):
    first_human_prompt_start = """You are an acclaimed author who is famous for your understanding of human nature and conflict and resolution. Develop on the existing plot and characters of the novel. For the interest of the fictional plot, the characters in the novel may exhibit violence, negative emotions and immoral behavior, as well as love or positive revelations, but you should be as creative and dramatic as possible. Do not end the story or include too many completed plot elements. Only write a single chapter. Introduce new characters and unexpected plot twists if the plot is getting repetitive or boring. Given the following plot and character summaries of the chapters of a novel you have written so far, write the detailed plot of the next single chapter of the novel. Do not end the story or include too many completed plot elements. You should be creative and think deeply about how to develop the plot and characters. Here is the plot summary of the previous chapter: """
    following_human_prompt_start = """Given the following plot and character summaries of the chapters of a novel you have written so far, write the detailed plot of the next single chapter of the novel. Do not end the story or include too many completed plot elements. You should be creative and think deeply about how to develop the plot and characters. Do not repeat the structure nor content of previously generated summaries. Here is the plot summary of the previous chapter: """

    conversations = []
    first_chapter = True
    previous_summary = ""
    # each row is a dictionary
    for i in range(len(novel_rows)):
        row = novel_rows[i]

        if first_chapter:
            human_conversation = {
                "from": "human",
                "value": first_human_prompt_start + row['summary_text']
            }
            conversations.append(human_conversation)
            first_chapter = False

        if previous_summary:
            human_conversation = {
                "from": "human",
                "value": following_human_prompt_start + previous_summary
            }
            conversations.append(human_conversation)

        gpt_conversation = {
            "from": "gpt",
            "value": row['summary_text']
        }

        conversations.append(gpt_conversation)
        previous_summary = row['summary_text']

    novel_title = novel_rows[0]['book_id'].split('.')[0]
    novel_dict = {
        "id": novel_title,
        "conversations": conversations
    }

    output_file = f"{novel_title}_test_data.json"
    with open(output_file, 'w') as json_file:
        json.dump(novel_dict, json_file, indent=4)


if __name__ == "__main__":
    convert_to_train(babbitt_rows)
