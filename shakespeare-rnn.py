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

def back_propagation(inputs, wL,wS, b, activationFunc, learningRate, epochs, cIndex):
    for epoch in range(epochs):     
        for ind, rawInput in enumerate(inputs): #inp[input, output]
            inputOneHot = []
            for i in rawInput[0]:
                inputOneHot.append(convert_to_one_hot(i))
            inp = (inputOneHot,convert_to_one_hot(rawInput[1]))

            if(ind%200 == 0):
                print(ind)

            As ={} 
            dots = {}
            for layer in range(1, len(wL)-1):
                As[layer, 0] = np.zeros((len(wL[layer]), 1))
            for step in range(1, len(inp[0])+1): 
                As[(0, step)] = inp[0][step-1]
      
                for layer in range(1, len(wL)-1):
                    dots[(layer, step)] =wL[layer]@As[(layer-1, step)] + wS[layer] @As[(layer, step-1)] +b[layer]
                    As[(layer, step)] = activationFunc(dots[(layer, step)])

            #check dimensions should by like 38x0 no 128x1
            dots[(len(wL)-1, len(inp[0]))] =wL[len(wL)-1]@As[(len(wL)-2, len(inp[0]))]
            As[(len(wL)-1, len(inp[0]))] = softmax(dots[(len(wL)-1, len(inp[0]))])
            deltas= {}
        
            deltas[(len(wL)-1, len(inp[0]))] = softmax_derivative(As[(len(wL)-1, len(inp[0]))], inp[1])
            
            for layer in range(len(wL)-2,0,-1):
                deltas[(layer, len(inp[0]))] = activationFuncDerivative(dots[(layer, len(inp[0]))])*(np.transpose(wL[layer+1])@deltas[(layer+1, len(inp[0]))])
            #dont use last layer not dense

            for step in range(len(inp[0])-1, 0,-1):
                deltas[(len(wL)-2, step)] = activationFuncDerivative(dots[len(wL)-2, step])*(np.transpose(wS[len(wL)-2])@deltas[len(wL)-2, step+1])
            
            #one  layer down
            for step in range(len(inp[0])-1, 0,-1):
                for layer in range(len(wL)-3,0,-1):
                    deltas[(layer, step)]=activationFuncDerivative(dots[(layer, step)])*(np.transpose(wS[layer])@deltas[(layer, step+1)] )+ activationFuncDerivative(dots[(layer, step)])*(np.transpose(wL[layer+1])@deltas[(layer+1, step)])

            #new sec same as dnn changing lr
            b[len(wL)-1] = b[len(wL)-1]+learningRate*deltas[(len(wL)-1,len(inp[0]))]
            wL[len(wL)-1] = wL[len(wL)-1]+learningRate*deltas[(len(wL)-1,len(inp[0]))] *As[(len(wL)-1,len(inp[0]))]

            #shift down 1
            for layer in range(1, len(wL)-1):
                biasSum = 0
                weightLSum = 0
                weightSSum = 0
                for step in range(1, len(inp[0])+1):
                    biasSum += deltas[(layer, step)] 
                    weightLSum += deltas[(layer, step)] @ np.transpose(As[(layer-1, step)])
                    if(step!=1): #except for first
                        weightSSum += deltas[(layer, step)] @ np.transpose(As[layer, step-1])
     
                wL[layer] = wL[layer]+(learningRate*weightLSum) 
                wS[layer] = wS[layer]+(learningRate*weightSSum)
                b[layer]=b[layer]+(learningRate*biasSum)     
    
            if((ind+cIndex)%25000 == 0):
                with open("w_b_current.pkl", "wb") as f:
                    pickle.dump((wL, wS, b, ind+cIndex), f)
                    print(f"current w/b filed save ({str(ind+cIndex)})")
                if((ind+cIndex)%100000 == 0):
                    with open(f"w_b_{str(ind+cIndex)}.pkl", "wb") as f:
                        pickle.dump((wL, wS, b), f)
                        print("time stamp file generated")
    return wS,wL, b


def create_rand_values(dimensions):
    weightStep= [None]
    biases = [None]
    weightsLayer=[None]
    for i in range(1,len(dimensions)-1):
        temp = dimensions[i-1]+2 * dimensions[i]
        r = (3/temp)**(0.5)

        weightStep.append(2*r*np.random.rand(dimensions[i],dimensions[i]) - r)
        weightsLayer.append(2*r*np.random.rand(dimensions[i],dimensions[i-1]) - r)
        biases.append(2*r*np.random.rand(dimensions[i],1)-r)
   
    temp = dimensions[-2]+ dimensions[-1]
    r = (3/temp)**(0.5)
    weightStep.append(2*r*np.random.rand(dimensions[-1],dimensions[-1]) - r)
    weightsLayer.append(2*r*np.random.rand(dimensions[-1],dimensions[-2]) - r)
    biases.append(2*r*np.random.rand(dimensions[-1],1)-r)
        
    return weightStep,weightsLayer,biases

with open("shakespeare-tests.pkl", "rb") as f:
    data = pickle.load(f)    
    with open("w_b_current.pkl", "rb") as f:
        w1L, w1S, b1,currIdx = pickle.load(f)
    back_propagation(data, w1L, w1S, b1, np.tanh, 0.01, 5, currIdx)

