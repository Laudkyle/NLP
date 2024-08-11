import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# Loading the text 
text = open('robert_frost.txt', 'rb').read().decode(encoding="utf-8").lower()



characters = sorted(set(text))

char_to_index = dict((c,i) for i,c in enumerate(characters))
index_to_char = dict((i,c) for i,c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 5

sentences = []
next_char = []
 
for i in range(0, len(text)-SEQ_LENGTH,STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_char.append(text[i+SEQ_LENGTH])


x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i,t, char_to_index[character]] =1
    y[i, char_to_index[next_char[i]]] = 1
    


model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer = RMSprop(0.01))
model.fit(x, y, epochs=50, batch_size = 256)
model.save('texter.h5')

model = tf.keras.models.load_model('texter.h5')

def sample(preds, temp):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temp
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temp):
    start_index = random.randint(0, len(text)- SEQ_LENGTH -1 )
    generated =''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x= np.zeros((1,SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0,t, char_to_index[character]] = 1
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temp)
        next_char =index_to_char[next_index]
        
        generated += next_char
        
        sentence = sentence[1:] + next_char
    return generated
print('----------------0.2------------------------')    
print(generate_text(100,0.2))
print('------------------0.4----------------------')    
print(generate_text(100,0.4))
print('------------------0.6----------------------')    
print(generate_text(300,0.6))
print('-------------------0.8---------------------')    
print(generate_text(100,0.8))