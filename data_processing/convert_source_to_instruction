import json

def convert_data_format(data):
    converted_data = []
    question_id = 0
    for novel_data in data:
        category = "writing"
        turns = []

        system_prompt = None
        user_prompt = None

        for conv in novel_data["conversations"]:
            if conv["role"] == "system":
                system_prompt = conv["content"]
            elif conv["role"] == "user":
                if system_prompt is not None:
                    turns.append(system_prompt)
                    turns.append(conv["content"])
                    system_prompt = None
                else:
                    turns.append(conv["content"])

        converted_data.append({
            "question_id": question_id,
            "category": category,
            "turns": turns
        })
        question_id += 1

    return converted_data

def main():
    with open("sources.json", "r") as f:
        data = json.load(f)

    converted_data = convert_data_format(data)

    with open("converted_sources.json", "w") as f:
        json.dump(converted_data, f, indent=2)

if __name__ == "__main__":
    main()
