- first question why pytorch when we have numpy?
yes we can use numpy and do the things there is no issue. but we would have to manually implement the backward-pass calculations. 
PyTorch provides automatic differentiation through Autograd, making gradient computation and neural-network training much easier. 
It can also let us use GPU instead of CPU. 

**corething to remember:**

B / batch_size
    = How many sequences we're processing together.

T / context_length/sequence length
    = How many token positions are in each sequence.

C / embedding_dim / d_model
    = How many numbers represent each token.

# Tensor Manupulation 

## 1D tensors
x = torch.tensor([1,2,3])
print(x,"\n",x.shape)
shape will be size([3])
it is a 1D tensor

## 2D tensors
x = torch.tensor([[1],[2],[3]])
print(x,"\n",x.shape)
shape will be size([3,1])

x = torch.tensor([[1,2],[3,4]])
print(x,"\n",x.shape)
shape will be size([2,2])


## tensor indexing 

all nonsense will be there. but we mainly use 

<!-- to get entire row -->
print(x[1,:])
print(x[0,:])

<!-- to get entire column -->
: means take everthing in that like x[:,1] -> means every row and column 1
print(x[:,1])

## reshape
if originally the shape is (100) eg like a 1D tensor
now you want to reshape it . one important thing is while reshaping no of elements should remain same. so we can reshape in many different ways eg:
x.reshape(4,25)
x.reshape(10,10)
x.reshape(5,20) ...

we commonly see torch.reshape(-1) -> it means we flattening the things like making it a 1D tensor. 
if we sure with any dim we can use other as -1 so that pytorch will calculate and give. eg:

x.reshape(20,-1) -> it will automatically make it (20,5)

## unsequeeze and sequeeze

y = x.unsqueeze(0) - it means we adding another dim at position 0 
suppose shape is (4,12) - it will become (1,4,12)

similar we can sequeeze() - it remove any dim of size 1. 
suppose shape is (1,4,23,1,3,1) - it will become (4,23,3)


## transpose
<!-- core thing transpose(0,1) means we are swapping 0 dimension with 1 and 1 dimension with 0 -->
if x shape is [1,3,1,2]
y = x_tensor.transpose(2,3) # means we are swapping 2 and 3 dimensions so output of shape will be [1,3,2,1]


## permute

y = x_tensor.permute(1,0,3,2)
suppose our shape is [8,256,6,64]
lets you rearrange all dimensions in any order
y shape will be [256,8,64,6]

## using dim
generally in transformers we have shape like 
[batch, tokens, embedding]
dim=0 → across batches
dim=1 → across tokens
dim=2 or dim=-1 → across embedding features

so whenever we do something like 
probs = torch.softmax(logits, dim=-1)
that means we doing softmax for the last dim i.e logits.. (in transformer architecture)




## broadcasting 
we compare from rightside and works only if end dimension matches or either is 1 or no dim
eg:

# we compare from rightside and works only if end dimension matches or either is 1
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
]) - x shape is (2,3)
b = torch.tensor([10, 20, 30]) shape is (3)
z = x+b this works because end shape matches.. 
so z will be 
[   [11, 22, 33],
    [14, 25, 36]
]

or b = torch.tensor([10]) then it shape will be (1)
then also it will work.

we can also do like 
b = 10
and do z = x+b that also will work. like it effects every element. 

see if a shape is (2,3) and b shape is (3,1) do you this a+b works no because dimensions must match or else any one can be 1. 

we need to check from right -> left. 
2 3
3 1

2 and 3 only not matching so it wont work.

## multiplication

there are 2 things in multiplication '@' and '*' 
suppose 
A = [[a1,b1],[c1,d1]]
B = [[a2,b2],[c2,d2]]

@ means matrix multiplication
A @ B means 
[[a1*a2 + b1*c2,    a1*b2 + b1*d2],
 [c1*a2 + d1*c2,    c1*b2 + d1*d2]]


* means element wise multiplication
and A * B means 

[[a1*a2, b1*b2],
 [c1*c2, d1*d2]]

* means it uses element wise mutliplication plus broadcasting too. 

## cat

Joins tensors along an EXISTING dimension.

Does NOT create a new dimension.

[2,3] + [2,3]

cat(dim=0) → [4,3]
cat(dim=1) → [2,6]

## stack

Joins tensors by creating a NEW dimension.

Number of dimensions increases by 1.

Three tensors:
all tensors must have exactly the same shape.
[4,7]
[4,7]
[4,7]

stack(dim=0) → [3,4,7]
stack(dim=1) → [4,3,7]
stack(dim=2) → [4,7,3]



# Dataset and DataLoader
There are mainly 2 concepts:

1. **Dataset**
2. **DataLoader**

from torch.utils.data import Dataset, DataLoader

## Dataset

A `Dataset` represents our collection of **individual training examples**.

For example, suppose our dataset contains 100 sequences:

sequence 0
sequence 1
sequence 2
...
sequence 99

Then:
len(dataset)
would return -> 100

The important thing is:

> **Dataset decides what one training example looks like.**

For language models, one example might contain:

input  = [token1, token2, token3, token4]
target = [token2, token3, token4, token5]

If we are creating sequences from a long list of tokens using sliding windows, that sequence creation happens when constructing the **Dataset**.

For example:

tokens = 100
context_length = 10
stride = 1

Number of windows:
(100 - 10) / 1 + 1 = 91

So sliding/overlapping windows are related to **how we create the Dataset**, not the DataLoader's `batch_size`.


## DataLoader

Once we have a Dataset, the `DataLoader` takes the individual examples and **groups them into batches**.

```python
loader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True
)
```

Suppose:
len(dataset) = 100
batch_size = 10

Then each batch contains 10 examples:
100/10 -> 10 batches

We can iterate through them:

```python
for x, y in loader:
    print(x.shape)
    print(y.shape)
```

If each sequence has `context_length = 20`, we may get:

x.shape = [10, 20]
y.shape = [10, 20]
           ↑   ↑
           B   T

where:

B = batch size
T = context/sequence length

So the core idea is:

Raw data
   ↓
Dataset
   ↓
creates/represents individual training examples
   ↓
DataLoader
   ↓
groups those examples into batches
   ↓
Neural Network


# Maths & Stats

we already know matrix multiplication like @ and *. 
we also have dot product which is simply element wise multiplication + adding them 
a = [1,2,3] b = [4,5,6]
torch.dot(a,b) 
1*4+2*5+3*6 = 32
output is tensor(32)

what does this dot product says.. 
+ve -> generally pointing in similar directions
0 -> perpendicular 
-ve -> generally pointing in opposing directions

even though we have dot product we chose cosine similarity because when comparing how similar 2 vectors are if we do dot product it is influenced by magnitude of vectors. 
so we chose cosine similarity 

import torch
import torch.nn.functional as F

a = torch.tensor([1., 1.])
b = torch.tensor([2., 2.])
c = torch.tensor([100., 100.])

print(torch.dot(a, b)) ->4
print(torch.dot(a, c)) -> 200

print(F.cosine_similarity(a, b, dim=0)) -> 1
print(F.cosine_similarity(a, c, dim=0)) -> 1



## mean
mean = sum of values / number of values
the thing in pytorch we can do mean across dim
like 
x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])

x.shape - (2,3)

x.mean(dim = 0) means we are saying collpase dim =0 by taking mean
mean = 1+4/2,2+5/2,3+6/2 i.e (2.5,3.5,4.5)
shape is (3)
x.mean(dim =1) means we are saying collapse dimension 1 by taking mean
mean = 1+2+3/3,4+5+6/3 i.e (2,5)
shape is (2)

if you see x.shape = [32, 100, 768]
now 
y = x.mean(dim=-1)
that means taking all 768 embeddings take mean and collpase that dimension
so 
y shape become (32,100)

## varience 
it says How spread out are the values around the mean?
Variance = mean((x - mean)²)

x = torch.tensor([
    [1., 2., 3.],
    [10., 20., 30.]
])
here x shape is (2,3)
x.var(dim = -1,correction = 0)
it is asking taking varience in dim = -1 and collpase that
so we calculate varience for [1,2,3] and [10,20,30] seperately 
[0.667, 66.667]
now we reduced that to shape (2)

there is one annoyed thing 
If our original values are measured in some unit x, variance is measured in x² because we squared the differences.

so we introduce 

## standard deviation 
standard deviation = √variance
in pytorch we use 
x.std(correction = 0)


## standardization/normalization
(x - mean) / std

## softmax
Neural network -> logits -> softmax -> probabilities
generally the logits are not probability. like if those probabilities we need to get 1 by adding all of them. to make them probabilities we use softmax. 

probs = torch.softmax(logits, dim=-1)

softmax(xᵢ) = eˣⁱ / Σeˣʲ
 