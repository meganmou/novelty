from datasets import load_dataset
import json

dataset = load_dataset("kmfoda/booksum")
train_dataset = dataset["train"]


little_women_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Little Women") and row["source"] == "sparknotes")
a_tale_of_two_cities_rows = train_dataset.filter(lambda row: row["book_id"].startswith("A Tale of Two Cities") and row["source"] == "cliffnotes")
emma_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Emma") and row["source"] == "sparknotes")
alice_in_wonderland_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Alice in Wonderland") and row["source"] == "sparknotes")
persuasion_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Persuasion") and row["source"] == "sparknotes")
gullivers_travels_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Gulliver's Travels") and row["source"] == "cliffnotes")
mansfield_park_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Mansfield Park") and row["source"] == "sparknotes")
treasure_island_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Treasure Island") and row["source"] == "cliffnotes")
david_copperfield_rows = train_dataset.filter(lambda row: row["book_id"].startswith("David Copperfield") and row["source"] == "cliffnotes")
dr_jekyll_mr_hyde_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Dr. Jekyll and Mr. Hyde") and row["source"] == "cliffnotes")

oliver_twist_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Oliver Twist") and row["source"] == "cliffnotes")
# sense_and_sensibility_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Sense and Sensibility") and row["source"] == "cliffnotes")
# tess_of_durbervilles_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Tess of the d'Urbervilles") and row["source"] == "cliffnotes")
# brothers_karamazov_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Brothers Karamazov") and row["source"] == "cliffnotes")
deerslayer_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Deerslayer") and row["source"] == "cliffnotes")
house_of_seven_gables_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The House of the Seven Gables") and row["source"] == "cliffnotes")
last_of_mohicans_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Last of the Mohicans") and row["source"] == "cliffnotes")
# picture_of_dorian_gray_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Picture of Dorian Gray") and row["source"] == "cliffnotes")
portrait_of_a_lady_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Portrait of a Lady") and row["source"] == "cliffnotes")
# prince_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Prince") and row["source"] == "cliffnotes")
# red_and_black_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Red and the Black") and row["source"] == "cliffnotes")
# taming_of_shrew_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Taming of the Shrew") and row["source"] == "cliffnotes")
# tempest_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Tempest") and row["source"] == "cliffnotes")
vanity_fair_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Vanity Fair") and row["source"] == "cliffnotes")
winesburg_ohio_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Winesburg Ohio") and row["source"] == "cliffnotes")
main_street_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Main Street") and row["source"] == "cliffnotes")

three_musketeers_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Three Musketeers") and row["source"] == "sparknotes")
king_lear_rows = train_dataset.filter(lambda row: row["book_id"].startswith("King Lear") and row["source"] == "sparknotes")
cyrano_de_bergerac_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Cyrano de Bergerac") and row["source"] == "sparknotes")
romeo_and_juliet_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Romeo and Juliet") and row["source"] == "cliffnotes")
white_fang_rows = train_dataset.filter(lambda row: row["book_id"].startswith("White Fang") and row["source"] == "sparknotes")
tartuffe_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Tartuffe") and row["source"] == "cliffnotes")
pygmalion_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Pygmalion") and row["source"] == "cliffnotes")
julius_caesar_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Julius Caesar") and row["source"] == "cliffnotes")
incidents_in_life_of_slave_girl_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Incidents in the Life of a Slave Girl") and row["source"] == "cliffnotes")
mill_on_floss_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Mill on the Floss") and row["source"] == "cliffnotes")
red_badge_of_courage_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Red Badge of Courage") and row["source"] == "sparknotes")
regeneration_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Regeneration") and row["source"] == "sparknotes")
anne_of_green_gables_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Anne of Green Gables") and row["source"] == "sparknotes")
pickwick_papers_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Pickwick Papers") and row["source"] == "cliffnotes")
howards_end_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Howards End") and row["source"] == "sparknotes")
green_mansions_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Green Mansions") and row["source"] == "cliffnotes")
frederick_douglass_narrative_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Narrative of the Life of Frederick Douglass: An American Slave") and row["source"] == "cliffnotes")
northanger_abbey_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Northanger Abbey") and row["source"] == "sparknotes")
paradise_lost_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Paradise Lost") and row["source"] == "cliffnotes")
turn_of_screw_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Turn of the Screw") and row["source"] == "sparknotes")
candide_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Candide") and row["source"] == "sparknotes")
kidnapped_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Kidnapped") and row["source"] == "sparknotes")
sister_carrie_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Sister Carrie") and row["source"] == "sparknotes")
siddhartha_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Siddhartha") and row["source"] == "cliffnotes")
jude_the_obscure_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Jude the Obscure") and row["source"] == "cliffnotes")
sons_and_lovers_rows = train_dataset.filter(lambda row: row["book_id"].startswith("Sons and Lovers") and row["source"] == "cliffnotes")
house_of_mirth_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The House of Mirth") and row["source"] == "sparknotes")
scarlet_letter_rows = train_dataset.filter(lambda row: row["book_id"].startswith("The Scarlet Letter") and row["source"] == "sparknotes")

training_data = [little_women_rows, a_tale_of_two_cities_rows, emma_rows, alice_in_wonderland_rows, 
                 persuasion_rows, gullivers_travels_rows, mansfield_park_rows, treasure_island_rows, 
                 david_copperfield_rows, dr_jekyll_mr_hyde_rows, oliver_twist_rows, deerslayer_rows, 
                 house_of_seven_gables_rows, last_of_mohicans_rows, portrait_of_a_lady_rows, 
                 vanity_fair_rows, winesburg_ohio_rows, three_musketeers_rows, king_lear_rows, 
                 cyrano_de_bergerac_rows, romeo_and_juliet_rows, white_fang_rows, tartuffe_rows, 
                 pygmalion_rows, julius_caesar_rows, incidents_in_life_of_slave_girl_rows, 
                 mill_on_floss_rows, red_badge_of_courage_rows, regeneration_rows, anne_of_green_gables_rows, 
                 pickwick_papers_rows, howards_end_rows, green_mansions_rows, frederick_douglass_narrative_rows, 
                 northanger_abbey_rows, paradise_lost_rows, turn_of_screw_rows, candide_rows, kidnapped_rows, 
                 sister_carrie_rows, siddhartha_rows, jude_the_obscure_rows, main_street_rows, house_of_mirth_rows, 
                 sons_and_lovers_rows, scarlet_letter_rows]

all_training_data = []
def convert_to_train(novel_rows):
    first_human_prompt_start = """You are an acclaimed author who is famous for your understanding of human nature and conflict and resolution. Develop on the existing plot and characters of the novel. For the interest of the fictional plot, the characters in the novel may exhibit violence, negative emotions and immoral behavior, as well as love or positive revelations, but you should be as creative and dramatic as possible. Do not end the story or include too many completed plot elements. Only write a single chapter. Introduce new characters and unexpected plot twists if the plot is getting repetitive or boring. Given the following plot and character summaries of the chapters of a novel you have written so far, write the detailed plot of the next single chapter of the novel. Do not end the story or include too many completed plot elements. You should be creative and think deeply about how to develop the plot and characters. Here is the plot summary of the previous chapter: """
    following_human_prompt_start = """Given the plot and character summaries of the chapters of a novel you have written so far, write the detailed plot of the next single chapter of the novel. Do not end the story or include too many completed plot elements. You should be creative and think deeply about how to develop the plot and characters. Do not repeat the structure nor content of previously generated summaries."""

    # conversations = []
    first_chapter = True
    previous_summary = ""
    # each row is a dictionary
    for i in range(len(novel_rows)):
        row = novel_rows[i]

        conversations = []
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
                "value": following_human_prompt_start
            }
            conversations.append(human_conversation)

        gpt_conversation = {
            "from": "gpt",
            "value": row['summary_text']
        }
        conversations.append(gpt_conversation)
        previous_summary = row['summary_text']

        

        novel_title = novel_rows[0]['book_id'].split('.')[0] + str(i)
        novel_dict = {
        "id": novel_title,
        "conversations": conversations
        }
        all_training_data.append(novel_dict)

    output_file = "individual_chapter_training_data.json"
    with open(output_file, 'w') as json_file:
        json.dump(all_training_data, json_file, indent=4)


if __name__ == "__main__":
    for novel_rows in training_data:
        if len(novel_rows) > 0:
            convert_to_train(novel_rows)
