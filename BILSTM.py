import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.optimizers import RMSprop
# ---------------------------- load the dataset ----------------------------

with open("sherlock.txt", "r") as file:
    lines = file.readlines()
data = pd.DataFrame(lines, columns=[None])
pd.set_option("max_colwidth", None)
print(data)


# ---------------------------- data cleaning ----------------------------

data = data.iloc[:11874]
def remove_chars(dataset):
    remove_n = re.sub(r'\n', ' ', dataset)
    cleaned_dataset = re.sub(r'_', '', remove_n)
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
    cleaned_sentence_4 = re.sub(r'â€', '"', cleaned_sentence_3)
    cleaned_sentence = re.sub(r'â€˜', "'", cleaned_sentence_4)
    cleaned_sentence = cleaned_sentence.lower()

    return cleaned_sentence

data = data.apply(clean_sentence)



print(data.head(20))
print(data.iloc[39])
print(data.iloc[63])
print(data.iloc[75])


# ---------------------------- tokenize ----------------------------

tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1
print("Total number of words: ", total_words)
print("<oov>: ", tokenizer.word_index['<oov>'])
print("holmes: ", tokenizer.word_index['holmes'])
print("i: ", tokenizer.word_index['i'])
print("he: ", tokenizer.word_index['he'])


# ---------------------------- n_gram ----------------------------

input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    print(token_list)

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

print(input_sequences)
print("Total input sequences: ", len(input_sequences))


# ---------------------------- padding ----------------------------

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))
input_sequences[5]


# ---------------------------- feature selection ----------------------------

xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


# ---------------------------- build and fit model: BiLSTM ---------------------------

model = Sequential()
model.add(Embedding(total_words, 150, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Bidirectional(LSTM(200)))
model.add(Dense(total_words, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer= RMSprop(learning_rate=0.01), metrics=['accuracy'])
history = model.fit(xs, ys, epochs=1, verbose=1)


# ---------------------------- evaluate ---------------------------

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


# ---------------------------- predict ---------------------------

seed_text = "holmes and i"
next_words = 2

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
