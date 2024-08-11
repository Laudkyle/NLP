import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the data
with open('chat_data.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Prepare inputs and outputs
contexts = df['context'].tolist()
questions = df['question'].tolist()
responses = df['response'].tolist()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(contexts + questions + responses)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
def texts_to_sequences(texts):
    return tokenizer.texts_to_sequences(texts)

contexts_seq = texts_to_sequences(contexts)
questions_seq = texts_to_sequences(questions)
responses_seq = texts_to_sequences(responses)

# Combine context and question
combined_sequences = [ctx + q for ctx, q in zip(contexts_seq, questions_seq)]

# Determine max lengths
max_len_combined = max(len(seq) for seq in combined_sequences)
max_len_response = max(len(seq) for seq in responses_seq)

# Pad sequences
combined_sequences_padded = pad_sequences(combined_sequences, maxlen=max_len_combined, padding='post')
responses_padded = pad_sequences(responses_seq, maxlen=max_len_response, padding='post')

# Convert responses to categorical
responses_categorical = [to_categorical(seq, num_classes=vocab_size) for seq in responses_padded]

# Ensure padding length matches the output length
max_len_combined = combined_sequences_padded.shape[1]
max_len_response = responses_padded.shape[1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(combined_sequences_padded, responses_categorical, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len_combined))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))
