import pickle
import numpy as np
from random import shuffle

text = open("shakespeare.txt").read()
frequency = {}
for i in text:
    if(i.lower() in frequency):
        frequency[i.lower()] +=1
    else:
        frequency[i.lower()] = 1
oneHotVectors={k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
print(oneHotVectors)
oneHotVectorsKeys = []
for i in oneHotVectors:
    oneHotVectorsKeys.append(i)

def convert_to_one_hot(keys, char):
    inp = np.zeros((len(keys), 1))
    inp[keys.index(char),0] =1
    return inp
print(convert_to_one_hot(oneHotVectorsKeys, "a"))

cutLines = []
for i in range(0, len(text)-101):
    cutLines.append((text[i:i+101].lower(), text[i+101].lower()))
shuffle(cutLines)
with open("shakespeare-tests.pkl", "wb") as f:
    pickle.dump(cutLines,f)

with open("shakespeare-tests.pkl", "rb") as f:
    data = pickle.load(f)
    for i in range(4):
        print(data[i])

