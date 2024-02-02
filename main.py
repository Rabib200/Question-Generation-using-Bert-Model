
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def get_distractors_of_a_word(syn,word):
    distractors = []
    word = word.lower()
    orig_word = word

    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


original_word = "Blue"

syns = wn.synsets(original_word,'n')
for syn in syns:
    print(syn, ": ",syn.definition(),'\n')

synset_to_use = wn.synsets(original_word,'n')[0]

print("Original word: ",original_word.capitalize())
print("Distractors: ")
print(get_distractors_of_a_word(synset_to_use,original_word))