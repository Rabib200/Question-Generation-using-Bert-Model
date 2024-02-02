import re
import torch
from tabulate import tabulate
from nltk.corpus import wordnet as wn

from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer
import time


MAX_SEQ_LENGTH = 128

def get_sense(sent):
  re_result = re.search(r"\[TGT\](.*)\[TGT\]", sent)
  if re_result is None:
      print("\nIncorrect input format. Please try again.")

  ambiguous_word = re_result.group(1).strip()

  results = dict()

  wn_pos = wn.NOUN
  for i, synset in enumerate(set(wn.synsets(ambiguous_word, pos=wn_pos))):
      results[synset] =  synset.definition()

  if len(results) ==0:
    return (None,None,ambiguous_word)

  # print (results)
  sense_keys=[]
  definitions=[]
  for sense_key, definition in results.items():
      sense_keys.append(sense_key)
      definitions.append(definition)


  record = GlossSelectionRecord("test", sent, sense_keys, definitions, [-1])

  features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_token_segment_id=0,
                                            disable_progress_bar=True)[0]

  with torch.no_grad():
      logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
      # for i, bert_input in tqdm(list(enumerate(features)), desc="Progress"):
      for i, bert_input in list(enumerate(features)):
          logits[i] = model.ranking_linear(
              model.bert(
                  input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
              )[1]
          )
      scores = softmax(logits, dim=0)

      preds = (sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True))


  # print (preds)
  sense = preds[0][0]
  meaning = preds[0][1]
  return (sense,meaning,ambiguous_word)


sentence1 = "Srivatsan loves to watch **cricket** during his free time"


sentence_for_bert = sentence1.replace("**"," [TGT] ")
sentence_for_bert = " ".join(sentence_for_bert.split())
sense,meaning,answer = get_sense(sentence_for_bert)

print (sentence1)
print (sense)
print (meaning)

sentence2 = "Srivatsan is annoyed by a **cricket** in his room"
sentence_for_bert = sentence2.replace("**"," [TGT] ")
sentence_for_bert = " ".join(sentence_for_bert.split())
sense,meaning,answer = get_sense(sentence_for_bert)

print ("\n-------------------------------")
print (sentence2)
print (sense)
print (meaning)
