from transformers import T5ForConditionalGeneration,T5Tokenizer

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

def get_question(sentence,answer):
  text = "context: {} answer: {} </s>".format(sentence,answer)
  print (text)
  max_len = 256
  encoding = question_tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = question_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=200)


  dec = [question_tokenizer.decode(ids) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question


sentence1 = "Rabib is annoyed by the **cricket** insect in his room"
sentence2 = "Rabib loves to play **cricket**"

answer = "cricket"

sentence_for_T5 = sentence1.replace("**"," ")
sentence_for_T5 = " ".join(sentence_for_T5.split()) 
ques = get_question(sentence_for_T5,answer)
print (sentence1)
print (ques)


print ("\n**************************************\n")
sentence_for_T5 = sentence2.replace("**"," ")
sentence_for_T5 = " ".join(sentence_for_T5.split()) 
ques = get_question(sentence_for_T5,answer)
print (sentence2)
print (ques)