import numpy as np
import pickle

from random import shuffle

text = open("shakespeare.txt").read()
frequency = {}
for i in text:
    if(i.lower() in frequency):
        frequency[i.lower()] +=1
    else:
        frequency[i.lower()] = 1
oneHotVectors={k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
# print(oneHotVectors)
oneHotVectorsKeys = []
for i in oneHotVectors:
    oneHotVectorsKeys.append(i)

def convert_to_one_hot(char):
    inp = np.zeros((len(oneHotVectorsKeys), 1))
    inp[oneHotVectorsKeys.index(char),0] =1
    return inp


cutLines = []
for i in range(0, len(text)-101):
    cutLines.append((text[i:i+101].lower(), text[i+101].lower()))
shuffle(cutLines)
with open("shakespeare-tests.pkl", "wb") as f:
    pickle.dump(cutLines,f)





def activationFuncDerivative(x):
    return 1 / np.cosh(x) ** 2

def softmax(dot_x):
    temp = np.exp(dot_x)
    return temp/np.sum(temp)
def softmax_derivative(x, y):
    return (np.eye(y.shape[0]) + (-1 * x)) @y

def generate(wL,wS, b, activationFunc, letters):   
    inp = []
    for i in letters:
        inp.append(convert_to_one_hot(i))

    As ={} 
    dots = {}
    for layer in range(1, len(wL)-1):
        As[layer, 0] = np.zeros((len(wL[layer]), 1))
    for step in range(1, len(inp)+1): 
        As[(0, step)] = inp[step-1]

        for layer in range(1, len(wL)-1):
            dots[(layer, step)] =wL[layer]@As[(layer-1, step)] + wS[layer] @As[(layer, step-1)] +b[layer]
            As[(layer, step)] = activationFunc(dots[(layer, step)])

    #check dimensions should by like 38x0 no 128x1
    dots[(len(wL)-1, len(inp))] =wL[len(wL)-1]@As[(len(wL)-2, len(inp))]
    As[(len(wL)-1, len(inp))] = softmax(dots[(len(wL)-1, len(inp))])
    print(As)
    maxI = 0
    for i in range(39):
        if(As[(len(wL)-1, len(inp))][i,0] > As[(len(wL)-1, len(inp))][maxI, 0]):
            maxI = i
    print(oneHotVectorsKeys[maxI])

with open("shakespeare-tests.pkl", "rb") as f:
    with open("w_b_current.pkl", "rb") as f:
        w1L, w1S, b1, currIdx = pickle.load(f)
    generate(w1L, w1S, b1, np.tanh, "First Citizen: Before we proceed.".lower())

print(oneHotVectorsKeys)