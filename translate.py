from transformers import T5Tokenizer, T5ForConditionalGeneration


def main(text, langueEntry, langueExit):

    tokenizers = T5Tokenizer.from_pretrained('t5-large')
    model = T5ForConditionalGeneration.from_pretrained('t5-base', torch_dtype="auto")


    input_ids = tokenizers("translate "+langueEntry+" to "+langueExit+" :"+text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, length_penalty=0.3, max_length=256)

    decoded = tokenizers.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return decoded