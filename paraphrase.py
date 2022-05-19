import transformers as tf
import torch as th
import sentencepiece as sp
import nltk
import time
import math
import langdetect as ld
from tqdm import tqdm

# import functions
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize


def main(text, type):

    if type=='URL':
        article = Article(text)
        article.download()
        article.parse()
        resText = article.text
    else:
        resText = text

    tokenizer = tf.AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = tf.AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


    nltk.download('punkt')


    def my_paraphrase(text):

        text = "paraphrase " + text + "</s>"
        encoding = tokenizer.encode_plus(text, padding=True, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 max_length=256,
                                 do_sample=True,
                                 top_k=120,
                                 top_p=0.95,
                                 early_stopping=True,
                                 num_return_sequences=1
                                 )

        output = tokenizer.decode(outputs[0],
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)

        return output

    output = " ".join([my_paraphrase(sent) for sent in tqdm(sent_tokenize(resText))])

    return output, ld.detect(resText)
