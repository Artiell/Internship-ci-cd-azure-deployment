import torch

from transformers import (
    AutoTokenizer,
AutoModelForSeq2SeqLM
)
import sentencepiece as spm

def main(text):

    pathh = "Ghani-25/SummAbsFR"
    model = AutoModelForSeq2SeqLM.from_pretrained(pathh)

    tokenizer = AutoTokenizer.from_pretrained(pathh)

    input_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=True)]
    )

    model.eval()

    predict = model.generate(input_ids, max_length=100)[0]
    result = tokenizer.decode(predict, skip_special_tokens=True)

    return result