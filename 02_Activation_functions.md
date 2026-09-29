**what are Activation functions**

We need activation functions because stacking only linear layers still gives us a linear function; activations introduce non-linearity so neural networks can learn complex relationships.

the things we can't solve every problem with linear functions stacked up on other linear functions and so on.. we need to introduce non-linearity to solve real world problems.. 

eg: take we need to find a function that solve |x|. ie mod(x)
if we take linear (1,2) and  linear(2,1)
h1 = x
h2 = -x

now second layer adds them 
ie.
h = h1+h2 = x-x = 0.

suppose if we take a non-linear function between them like 
linear(1,2) ReLu(2,2) Linear(2,1)
h1 = ReLu(x)
h2 = ReLu(-x)
if either x>0 or x<0
h = h1+h2 = x +0 or 0+x = x
so we create or we can say made neural network learn . by introducing non linearity 


**Types of activation functions**
there are some many but we dont want all of those. 
we take 
sigmoid,tanh,ReLU, Leaky ReLU, GELU,SiLU/Swish, Softmax

| Activation | Formula / idea | Common use |
|---|---|---|
| **ReLU** | \(\max(0,x)\) | Classic hidden layers, CNNs/MLPs |
| **Leaky ReLU** | small negative slope instead of 0 | ReLU alternative |
| **Sigmoid** | maps to \(0 \rightarrow 1\) | Binary probability output, gates |
| **Tanh** | maps to \(-1 \rightarrow 1\) | RNNs/LSTMs, older architectures |
| **GELU** | smooth gating based on input magnitude | Transformers, e.g. BERT/GPT-style models |
| **SiLU / Swish** | \(x\sigma(x)\) | Modern neural networks/LLMs |
| **Softmax** | converts vector to probability distribution | Classification, attention |


ReLU 
max(0,x)

Leaky ReLU
if x>0 -> x
if x<0 -> 0.01x

sigmoid 
very positive  - 0
0 - 0.5
very negative - 1

Tanh similar to sigmoid but 
very positive  1
0 - 0
very negative  -1

GELU 
make transisiton smoother compared to ReLU. 

