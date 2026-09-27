# Why PyTorch When We Have NumPy?

Technically, we **can** build neural networks using NumPy.

NumPy already gives us:
- tensors/arrays
- matrix multiplication
- mathematical operations
- reshaping and indexing

So why do we need PyTorch?

The main reason is **training**.

A neural network roughly works like:

```text
input
  ↓
forward pass
  ↓
prediction
  ↓
loss
  ↓
backward pass
  ↓
update parameters
```

With NumPy, we would have to manually calculate and implement many of the gradients required during the backward pass.

PyTorch gives us **Autograd**, which automatically computes these gradients.

For example:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2
y.backward()

print(x.grad)
```

Output:

```text
tensor(4.)
```

Because:

```text
y = x²

dy/dx = 2x

at x = 2:

dy/dx = 4
```

PyTorch calculated this gradient automatically.

Another major advantage is that PyTorch can run tensor operations and neural networks on **GPUs**, which can make large deep-learning workloads much faster.

So the core idea is:

```text
NumPy
→ great for numerical computation

PyTorch
→ numerical computation
→ automatic differentiation
→ neural-network modules
→ GPU acceleration
→ tools for training deep-learning models
```

---

# Core Tensor Dimensions

When working with transformers, we constantly see three dimensions:

```text
B = batch size
T = sequence length / context length
C = embedding dimension / d_model
```

A transformer tensor commonly has shape:

```text
[B, T, C]
```

For example:

```text
[32, 128, 768]
```

means:

```text
32  → sequences processed together
128 → tokens in each sequence
768 → numbers representing each token
```

So:

### B — Batch Size

How many sequences/examples are processed together.

```text
B = 32
```

means we are processing 32 sequences at once.

### T — Sequence Length / Context Length

How many token positions are present in each sequence.

```text
T = 128
```

means each sequence contains 128 token positions.

### C — Embedding Dimension / d_model

How many numbers are used to represent each token.

```text
C = 768
```

means every token is represented by a vector containing 768 numbers.

Therefore:

```text
[B, T, C]
     ↓
[32, 128, 768]
```

can be understood as:

```text
32 sequences
    ↓
128 tokens per sequence
    ↓
768 values per token
```

---

# Tensor Manipulation

## 1D Tensors

```python
x = torch.tensor([1, 2, 3])

print(x)
print(x.shape)
```

Shape:

```text
torch.Size([3])
```

This is a **1D tensor** containing 3 elements.

```text
[1, 2, 3]
 ↑  ↑  ↑
 3 elements
```

---

## 2D Tensors

Example:

```python
x = torch.tensor([
    [1],
    [2],
    [3]
])

print(x.shape)
```

Shape:

```text
torch.Size([3, 1])
```

That means:

```text
3 rows
1 column
```

Another example:

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])

print(x.shape)
```

Shape:

```text
torch.Size([2, 2])
```

So we have:

```text
2 rows
2 columns
```

---

# Tensor Indexing

Suppose:

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

Shape:

```text
[2, 3]
```

Think of it as:

```text
        columns
        0  1  2

row 0  [1, 2, 3]
row 1  [4, 5, 6]
```

## Selecting a Row

```python
x[0, :]
```

Output:

```text
tensor([1, 2, 3])
```

```python
x[1, :]
```

Output:

```text
tensor([4, 5, 6])
```

Here `:` means **take everything along this dimension**.

So:

```text
x[1, :]
  ↑  ↑
 row everything
  1  in columns
```

---

## Selecting a Column

```python
x[:, 1]
```

means:

```text
every row
column 1
```

Output:

```text
tensor([2, 5])
```

The important idea is:

```text
:
=
take everything in this dimension
```

---

# Reshape

`reshape()` changes the shape of a tensor without changing the number of elements.

Suppose:

```python
x.shape
```

is:

```text
[100]
```

There are 100 total elements.

Therefore, we can reshape it into any compatible shape containing 100 elements:

```python
x.reshape(4, 25)
x.reshape(10, 10)
x.reshape(5, 20)
```

because:

```text
4 × 25  = 100
10 × 10 = 100
5 × 20  = 100
```

The core rule is:

```text
number of elements before reshape
=
number of elements after reshape
```

---

## Using `-1`

PyTorch can automatically calculate one dimension for us.

Suppose:

```python
x.shape = [100]
```

and we write:

```python
x.reshape(20, -1)
```

PyTorch knows:

```text
20 × ? = 100
```

Therefore:

```text
? = 5
```

and the resulting shape becomes:

```text
[20, 5]
```

---

## Flattening

A common operation is:

```python
x.reshape(-1)
```

This converts all elements into a single 1D tensor.

For example:

```text
before:

[[1, 2],
 [3, 4]]

shape = [2, 2]
```

after:

```python
x.reshape(-1)
```

we get:

```text
[1, 2, 3, 4]

shape = [4]
```

---

# Unsqueeze and Squeeze

## `unsqueeze()`

`unsqueeze(dim)` adds a new dimension of size `1`.

Suppose:

```text
x.shape = [4, 12]
```

Then:

```python
y = x.unsqueeze(0)
```

gives:

```text
[1, 4, 12]
```

because we inserted a new dimension at position `0`.

Similarly:

```python
y = x.unsqueeze(1)
```

gives:

```text
[4, 1, 12]
```

The existing data does not change. We are only changing how the tensor is shaped.

---

## `squeeze()`

`squeeze()` removes dimensions whose size is `1`.

Suppose:

```text
x.shape = [1, 4, 23, 1, 3, 1]
```

Then:

```python
x.squeeze()
```

produces:

```text
[4, 23, 3]
```

because all dimensions of size `1` were removed.

You can also remove a specific dimension:

```python
x.squeeze(dim)
```

provided that dimension has size `1`.

---

# Transpose

`transpose(dim0, dim1)` swaps **two dimensions**.

Suppose:

```text
x.shape = [1, 3, 1, 2]
```

The dimension indices are:

```text
dimension:   0  1  2  3
shape:      [1, 3, 1, 2]
```

Now:

```python
y = x.transpose(2, 3)
```

means:

```text
swap dimension 2 with dimension 3
```

Therefore:

```text
before = [1, 3, 1, 2]
after  = [1, 3, 2, 1]
```

Core idea:

```text
transpose(a, b)
=
swap dimension a and dimension b
```

---

# Permute

`permute()` lets us rearrange **all dimensions** into any order.

Suppose:

```text
x.shape = [8, 256, 6, 64]
```

Dimension positions:

```text
0 → 8
1 → 256
2 → 6
3 → 64
```

Now:

```python
y = x.permute(1, 0, 3, 2)
```

We are asking for dimensions in this order:

```text
1, 0, 3, 2
```

Therefore:

```text
256, 8, 64, 6
```

and:

```text
y.shape = [256, 8, 64, 6]
```

The difference is:

```text
transpose()
→ swaps two dimensions

permute()
→ rearranges all dimensions
```

---

# Understanding `dim`

This is extremely important in PyTorch.

Suppose a transformer tensor has shape:

```text
[B, T, C]
```

For example:

```text
[32, 100, 768]
```

Then:

```text
dim=0 → batch dimension
dim=1 → token/sequence dimension
dim=2 → embedding dimension
```

We can also write:

```text
dim=-1
```

which means:

```text
the last dimension
```

So for:

```text
[32, 100, 768]
```

both:

```text
dim=2
```

and:

```text
dim=-1
```

refer to the `768` dimension.

For example:

```python
probs = torch.softmax(logits, dim=-1)
```

means:

> Apply softmax across the values in the last dimension.

Important:

```text
dim=-1
```

does **not** mean "logits."

It simply means **the last dimension of the tensor**.

---

# Broadcasting

Broadcasting allows PyTorch to perform operations between tensors with different but compatible shapes.

The easiest rule is:

> Compare dimensions from **right to left**.

Two dimensions are compatible when:

```text
they are equal

OR

one of them is 1

OR

one tensor does not have that dimension
```

Example:

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

Shape:

```text
x → [2, 3]
```

Now:

```python
b = torch.tensor([10, 20, 30])
```

Shape:

```text
b → [3]
```

Compare from the right:

```text
x → 2  3
b →    3
       ↑
     matches
```

Therefore:

```python
z = x + b
```

works.

Result:

```text
[
    [11, 22, 33],
    [14, 25, 36]
]
```

Conceptually, PyTorch behaves as though `b` were repeated:

```text
[10, 20, 30]
[10, 20, 30]
```

---

Another example:

```python
b = torch.tensor([10])
```

Shape:

```text
[1]
```

Compare:

```text
2  3
   1
```

`3` and `1` are compatible because one of them is `1`.

So this also works.

---

A scalar also broadcasts:

```python
z = x + 10
```

which effectively adds `10` to every element.

---

## Example That Does NOT Broadcast

Suppose:

```text
A.shape = [2, 3]
B.shape = [3, 1]
```

Compare from right to left:

```text
A → 2  3
B → 3  1
       ↑
     3 vs 1 ✓

    ↑
  2 vs 3 ✗
```

The second comparison fails.

Therefore:

```python
A + B
```

does not work.

---

# Multiplication

There are two operations that are easy to confuse:

```text
*
@
```

They mean very different things.

---

## `*` — Element-Wise Multiplication

Suppose:

```text
A = [[a1, b1],
     [c1, d1]]

B = [[a2, b2],
     [c2, d2]]
```

Then:

```python
A * B
```

produces:

```text
[
    [a1*a2, b1*b2],
    [c1*c2, d1*d2]
]
```

Each element is multiplied by the corresponding element.

`*` also supports **broadcasting**.

---

# `@` — Matrix Multiplication

`@` performs matrix multiplication.

For:

```text
A = [[a1, b1],
     [c1, d1]]

B = [[a2, b2],
     [c2, d2]]
```

Then:

```python
A @ B
```

produces:

```text
[
    [a1*a2 + b1*c2,  a1*b2 + b1*d2],
    [c1*a2 + d1*c2,  c1*b2 + d1*d2]
]
```

The important shape rule is:

```text
[m, n] @ [n, p]
        ↑     ↑
        must match
```

Result:

```text
[m, p]
```

For example:

```text
[4, 3] @ [3, 2]
```

produces:

```text
[4, 2]
```

---

# `cat`

`torch.cat()` joins tensors along an **existing dimension**.

It does **not** create a new dimension.

Suppose:

```text
A.shape = [2, 3]
B.shape = [2, 3]
```

Then:

```python
torch.cat([A, B], dim=0)
```

produces:

```text
[4, 3]
```

because we joined them along dimension `0`.

And:

```python
torch.cat([A, B], dim=1)
```

produces:

```text
[2, 6]
```

Core idea:

```text
cat
→ combine along an existing dimension
→ number of dimensions stays the same
```

---

# `stack`

`torch.stack()` joins tensors by creating a **new dimension**.

All tensors must have exactly the same shape.

Suppose we have three tensors:

```text
A → [4, 7]
B → [4, 7]
C → [4, 7]
```

Then:

```python
torch.stack([A, B, C], dim=0)
```

produces:

```text
[3, 4, 7]
```

Using:

```python
torch.stack([A, B, C], dim=1)
```

produces:

```text
[4, 3, 7]
```

And:

```python
torch.stack([A, B, C], dim=2)
```

produces:

```text
[4, 7, 3]
```

Core difference:

```text
cat
→ joins along an existing dimension
→ dimensions stay the same

stack
→ creates a new dimension
→ number of dimensions increases by 1
```

---

# Dataset and DataLoader

There are two important concepts:

```text
Dataset
DataLoader
```

They solve different problems.

---

## Dataset

A `Dataset` represents a collection of **individual training examples**.

Suppose our dataset contains 100 examples:

```text
example 0
example 1
example 2
...
example 99
```

Then:

```python
len(dataset)
```

would return:

```text
100
```

The core idea is:

> **Dataset defines what one training example looks like.**

For a language model, one example might be:

```text
input  = [token1, token2, token3, token4]
target = [token2, token3, token4, token5]
```

The target is shifted by one token because the model learns to predict the next token.

---

## Sliding Windows

Suppose we have:

```text
100 tokens
context_length = 10
stride = 1
```

A window might move like:

```text
tokens 0 → 9
tokens 1 → 10
tokens 2 → 11
...
```

The number of windows is:

```text
floor((N - context_length) / stride) + 1
```

So:

```text
floor((100 - 10) / 1) + 1
= 91
```

Therefore, we get:

```text
91 windows
```

Sliding/overlapping windows are related to **how we construct our dataset**.

They are different from the DataLoader's `batch_size`.

---

# DataLoader

Once we have individual examples, the `DataLoader` groups those examples into **batches**.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True
)
```

Suppose:

```text
len(dataset) = 100
batch_size   = 10
```

Then, assuming no examples are dropped:

```text
100 / 10 = 10 batches
```

Each batch contains 10 examples.

We can iterate through them:

```python
for x, y in loader:
    print(x.shape)
    print(y.shape)
```

If every example contains 20 tokens:

```text
x.shape = [10, 20]
y.shape = [10, 20]
```

which means:

```text
             B   T
             ↓   ↓
shape =    [10, 20]
```

where:

```text
B = batch size
T = sequence/context length
```

So the full picture is:

```text
Raw Data
   ↓
Dataset
   ↓
defines individual examples
   ↓
DataLoader
   ↓
groups examples into batches
   ↓
Neural Network
```

---

# Maths & Statistics

## Dot Product

For two vectors:

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
```

we can calculate:

```python
torch.dot(a, b)
```

The dot product is:

```text
(1×4) + (2×5) + (3×6)

= 4 + 10 + 18

= 32
```

So:

```text
dot product
=
element-wise multiplication
+
sum
```

---

## Geometric Meaning of Dot Product

The dot product contains information about both:

```text
direction
+
magnitude
```

Roughly:

```text
positive → angle between vectors is less than 90°
zero     → vectors are perpendicular
negative → angle between vectors is greater than 90°
```

But there is an important problem when using dot product directly as a measure of **directional similarity**:

> The dot product is affected by vector magnitude.

For example:

```python
import torch
import torch.nn.functional as F

a = torch.tensor([1., 1.])
b = torch.tensor([2., 2.])
c = torch.tensor([100., 100.])

print(torch.dot(a, b))
print(torch.dot(a, c))
```

Outputs:

```text
4
200
```

Even though `b` and `c` point in exactly the same direction as `a`, their dot products are dramatically different because their magnitudes are different.

---

# Cosine Similarity

Cosine similarity removes the effect of magnitude and focuses on the **angle between vectors**.

Conceptually:

```text
cosine similarity
=
dot product / (magnitude of a × magnitude of b)
```

Using PyTorch:

```python
print(F.cosine_similarity(a, b, dim=0))
print(F.cosine_similarity(a, c, dim=0))
```

Outputs:

```text
1
1
```

Both vectors point in exactly the same direction as `a`.

Generally:

```text
 1  → same direction
 0  → perpendicular
-1  → opposite direction
```

---

# Mean

Mean is:

```text
mean = sum of values / number of values
```

Suppose:

```python
x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])
```

Shape:

```text
[2, 3]
```

Now:

```python
x.mean(dim=0)
```

means:

> Take the mean across dimension `0` and collapse that dimension.

We calculate:

```text
(1 + 4) / 2 = 2.5
(2 + 5) / 2 = 3.5
(3 + 6) / 2 = 4.5
```

Result:

```text
[2.5, 3.5, 4.5]
```

Shape:

```text
[3]
```

---

Now:

```python
x.mean(dim=1)
```

calculates:

```text
(1 + 2 + 3) / 3 = 2
(4 + 5 + 6) / 3 = 5
```

Result:

```text
[2, 5]
```

Shape:

```text
[2]
```

A useful mental model is:

> **`dim` tells us which dimension we are reducing/collapsing.**

---

## Transformer Example

Suppose:

```text
x.shape = [32, 100, 768]
```

and we do:

```python
y = x.mean(dim=-1)
```

`dim=-1` refers to:

```text
768
```

So for every token of every sequence, we take the mean of its 768 features.

Therefore:

```text
before = [32, 100, 768]
                       ↓
                    collapse

after  = [32, 100]
```

---

# Variance

Variance tells us:

> **How spread out are the values around their mean?**

Conceptually:

```text
variance = mean((x - mean)²)
```

Suppose:

```python
x = torch.tensor([
    [1., 2., 3.],
    [10., 20., 30.]
])
```

Shape:

```text
[2, 3]
```

Now:

```python
x.var(dim=-1, correction=0)
```

means:

> Calculate variance across the last dimension and collapse that dimension.

So we independently calculate the variance of:

```text
[1, 2, 3]
```

and:

```text
[10, 20, 30]
```

Result approximately:

```text
[0.667, 66.667]
```

Shape:

```text
[2]
```

The second group has a much larger variance because its values are much more spread out.

---

# Standard Deviation

One issue with variance is that the differences were squared.

If the original values have units:

```text
x
```

then variance has units:

```text
x²
```

Standard deviation solves this by taking the square root:

```text
standard deviation = √variance
```

In PyTorch:

```python
x.std(correction=0)
```

Standard deviation is therefore expressed in the **same units as the original values**.

---

# Standardization

A common operation is:

```text
(x - mean) / standard_deviation
```

This is called **standardization** or a **z-score transformation**.

Conceptually:

```text
x
↓
subtract mean
↓
center values around 0
↓
divide by standard deviation
↓
scale according to their spread
```

After standardization, the values have approximately:

```text
mean = 0
standard deviation = 1
```

assuming the statistics are computed over the same dimension being standardized.

---

# Softmax

Neural networks often produce raw numbers called **logits**.

For example:

```text
logits = [2.0, 1.0, 0.1]
```

These are **not probabilities**.

Probabilities should satisfy:

```text
each value is between 0 and 1
```

and:

```text
sum of probabilities = 1
```

Softmax converts logits into such a probability distribution.

```python
probs = torch.softmax(logits, dim=-1)
```

The formula is:

```text
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

Conceptually:

```text
logits
   ↓
exponentiate
   ↓
divide each value by the total
   ↓
probabilities
```

For example:

```text
logits
[2.0, 1.0, 0.1]

        ↓ softmax

probabilities
[0.659, 0.242, 0.099]

        ↓

sum ≈ 1
```

In a language model, suppose the vocabulary size is `50,000`.

The logits for one token position might have shape:

```text
[50000]
```

Applying:

```python
torch.softmax(logits, dim=-1)
```

creates a probability distribution across those `50,000` vocabulary entries.

So the model can assign probabilities like:

```text
"the"   → 0.35
"cat"   → 0.20
"dog"   → 0.10
...
```

The core pipeline is:

```text
Neural Network
      ↓
    logits
      ↓
   softmax
      ↓
probability distribution
```