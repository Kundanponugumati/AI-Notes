
# Layers 
In simple terms it takes some input, process it and gives some output.
```
input -> computation -> output
```
```python 
layer = nn.Linear(3, 2)
x = torch.tensor([
    [1., 2., 3.]
])
y = layer(x)
```
here we are passing tensor x of shape (1,3). the layer is doing some computation and giving output as y and its shape is (1,2)

internally the linear layer performs 
```math
Y = XW^T + b
```

you see the layer contains trianable parameters
```
weights
bias
```
Not every layer has trianable parameters so has and some hasn't.  
Eg:
```
Layers / Modules

├── Parameterized
│   ├── nn.Linear
│   ├── nn.Embedding
│   └── nn.LayerNorm
│
└── No trainable parameters
    ├── nn.ReLU
    ├── nn.GELU
    └── nn.Dropout
```

## nn.Linear





