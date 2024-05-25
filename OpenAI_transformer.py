import pandas as pd
import re
from transformers import OpenAIGPTTokenizer, DataCollatorForLanguageModeling, TextDataset, OpenAIGPTLMHeadModel, Trainer, TrainingArguments, pipeline


# ---------------------------- load the dataset ----------------------------

with open("sherlock.txt", "r") as file:
    lines = file.readlines()
data = pd.DataFrame(lines, columns=[None])
pd.set_option("max_colwidth", None)
print(data)


# ---------------------------- data cleaning ----------------------------

data = data.iloc[:11874]
def remove_chars(dataset):
    cleaned_dataset = re.sub(r'\n', ' ', dataset)
    cleaned_dataset = re.sub(r'_', '', cleaned_dataset)
    return cleaned_dataset
data = data[None].apply(remove_chars)


def find_next_character(text):
    index = text.find("â€")
    if index != -1 and index + 2 < len(text):
        return text[index + 2]
    else:
        return None


find_next = data.apply(find_next_character)
find_next.unique()



def clean_sentence(dataset):
    cleaned_sentence_1 = re.sub(r'â€”', '—', dataset)
    cleaned_sentence_2 = re.sub(r'â€™', "'", cleaned_sentence_1)
    cleaned_sentence_3 = re.sub(r'â€œ', '"', cleaned_sentence_2)
    cleaned_sentence_4 = re.sub(r'â€ ', '"', cleaned_sentence_3)
    cleaned_sentence = re.sub(r'â€˜', "'", cleaned_sentence_4)
    cleaned_sentence = cleaned_sentence.lower()

    return cleaned_sentence

data = data.apply(clean_sentence)



print(data.head(20))
print(data.iloc[39])
print(data.iloc[63])
print(data.iloc[75])


# ---------------------------- Train and Test split  ----------------------------

train = data[0:int(0.8*len(data))]
test = data[int(0.8*len(data)):]

train=''.join(train)
test=''.join(test)

with open("train.txt", "w") as fp:
    fp.write(train)
with open("test.txt", "w") as fp:
    fp.write(test)


# ---------------------------- tokenization ----------------------------

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='train.txt',
    overwrite_cache=True,
    block_size=20)

test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='test.txt',
    overwrite_cache=True,
    block_size=20)


# ---------------------------- load pretrained model + fine tuning ----------------------------

model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
training_args = TrainingArguments(
    output_dir = 'gpt_model',
    overwrite_output_dir = True,
    per_device_train_batch_size = 3,
    per_device_eval_batch_size = 3,
    learning_rate = 5e-4,
    num_train_epochs = 3,
)


# ---------------------------- train + evaluate ----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


trainer.train()
trainer.save_model()
trainer.evaluate(test_dataset)


# ---------------------------- pipeline ----------------------------

generator = pipeline('text-generation', tokenizer='openai-gpt', model='gpt_model')
def predict_next():
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    generator = pipeline('text-generation', tokenizer='openai-gpt', model='gpt_model')
    while (True):
        text = input('Enter the text: ')
        length = len(tokenizer.encode(text, return_tensors='pt')[0])

        max_length = length + 1

        print('Next Word: ')
        print(generator(text, max_length=max_length)[0]['generated_text'].split(' ')[-1])
        print(generator(text, max_length=max_length, num_beams=5)[0]['generated_text'].split(' ')[-1])
        print(
            generator(text, max_length=max_length, do_sample=True, temperature=0.7)[0]['generated_text'].split(' ')[-1])
predict_next()
