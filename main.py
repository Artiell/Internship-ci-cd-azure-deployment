# import libraries
import streamlit as st
import transformers as tf
import torch as th
import sentencepiece as sp
import nltk
import time
import math
import langdetect as ld

# import functions
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize

st.write("# T5 model for paraphrasing")


inputType = st.radio(
    "What do you want for input ?", ('URL', 'Text Input'))

url_input = ""
text_input = ""
resText = ""

if inputType == 'URL':
    url_input = st.text_area("Algo TAL input Link", placeholder="Display your link here")


elif inputType == 'Text Input':
    resText = st.text_area("Algo TAL input text", placeholder="Display your text here")

if url_input != "":
    article = Article(url_input)
    article.download()
    article.parse()
    resText = article.text

tokenizer = tf.AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = tf.AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# mise en place du paraphrase
nltk.download('punkt')
my_bar = st.progress(0)
percent_complete = 0

def my_paraphrase(text):
    global percent_complete
    tailleTxt = len(text)
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

    for percent in range(round(tailleTxt*100/len(resText))):
        time.sleep(0.1)
        percent_complete = percent_complete + 1
        print(percent_complete)
        if(percent_complete > 100):
            percent_complete = 100

        my_bar.progress(percent_complete)

    output = tokenizer.decode(outputs[0],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    return output

if(resText!=""):
    st.write("## Input Text")
    with st.expander("Show input text"):
        st.write(resText)


output = " ".join([my_paraphrase(sent) for sent in sent_tokenize(resText)])

if(output != ""):
    if(percent_complete!=100):
        my_bar.progress(100)


if(resText!=""):
    st.write("## Output Text")
    with st.expander("Show output text"):
        st.write(output)

if(resText!=""):
    st.write("## Language Detection")
    st.write("This text is written in : " + ld.detect(resText))