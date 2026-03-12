# LibCN API Documentation

This document describes the public interfaces of **Lib Chest NeuralNetwork (LibCN)**.

LibCN is a header-only C++ neural network library designed for educational purposes and experimentation.

To use the entire library, simply include:

```
#include "lib_chest_nn.hpp"
```

All core headers are located inside the `nn/` directory but are automatically included by `lib_chest_nn.hpp`.

---

# Matrix

Defined in:

```
nn/matrix.hpp
```

## Overview

`Matrix<T>` is the core mathematical container used throughout LibCN.

It stores a two-dimensional matrix and provides basic linear algebra operations required for neural network computation.

Internally it uses:

```
std::vector<std::vector<T>>
```

---

## Template Requirements

The template type `T` must satisfy:

- `T + T`
- `T - T`
- `T * T`
- comparable using

```
> < >= <= == !=
```

and must be printable using:

```
std::cout << value
```

---

## Constructors

### Default constructor

```
Matrix()
```

Creates an empty matrix.

Properties:

```
h = 0
l = 0
```

Printing an empty matrix produces:

```
{ NULL }
```

---

### Shape constructor

```
Matrix(size_t h, size_t l)
```

Creates a matrix with specified dimensions.

Parameters

```
h   number of rows
l   number of columns
```

Elements are allocated but not initialized with specific values.

---

### From std::vector

```
Matrix(const std::vector<std::vector<T>>& a)
```

Constructs a matrix from an existing `std::vector<std::vector<T>>`.

Rules:

- Column count is determined by `a[0].size()`
- Rows longer than this length are truncated
- Rows shorter than this length remain partially uninitialized

---

### Initializer list

```
Matrix(std::initializer_list<std::initializer_list<T>> init)
```

Allows matrix initialization using list syntax.

Example:

```
Matrix<float> I{
    {1,0,0},
    {0,1,0},
    {0,0,1}
};
```

---

### Copy constructor

```
Matrix(const Matrix<T>& other)
```

Creates a **deep copy** of another matrix.

---

## Element Access

Matrix elements can be accessed directly using:

```
matrix[row][column]
```

Example:

```
a[1][2]
```

---

## Output

Matrices can be printed using `std::cout`.

Example output for a 3×3 identity matrix:

```
{ 1 0 0
  0 1 0
  0 0 1 }
```

---

## resize

```
void resize(size_t h, size_t l)
```

Changes matrix dimensions.

Behavior:

- overflowing elements are discarded
- newly created cells remain uninitialized

---

## transpose

```
Matrix<T> transpose()
```

Returns the transpose of the matrix.

---

## append

```
Matrix<T> append(const Matrix<T>& a, Direction d) const
```

Concatenates matrices.

Direction is defined by

```
enum class Direction
{
    Up,
    Down,
    Left,
    Right
};
```

Behavior:

- `Up` / `Down` require equal column counts
- `Left` / `Right` require equal row counts

If dimensions do not match, an **empty matrix** is returned.

---

## Operators

### Addition

```
Matrix + Matrix
Matrix += Matrix
```

Performs element-wise addition.

---

### Subtraction

```
Matrix - Matrix
Matrix -= Matrix
```

Performs element-wise subtraction.

---

### Multiplication

Two behaviors exist.

#### Matrix multiplication

```
Matrix * Matrix
Matrix *= Matrix
```

Performs standard matrix multiplication.

---

#### Scalar multiplication

```
Matrix * T
Matrix *= T
```

Performs element-wise scalar multiplication.

---

## Hadamard product

```
Matrix<T> hadamard(const Matrix<T>& other)
```

Performs element-wise multiplication.

---

## apply

```
Matrix<T> apply(function)
```

Applies a function element-wise to the matrix.

Accepts:

- function objects
- function pointers

---

# Layer

Defined in:

```
nn/layer.hpp
```

## Overview

`Layer<T>` represents a fully connected neural network layer.

Each layer contains:

- weight matrix `W`
- bias vector `b`
- activation function

---

## Constructors

### Default constructor

```
Layer()
```

Creates an empty layer.

---

### Layer(size_t i, size_t o)

```
Layer(size_t input_size, size_t output_size)
```

Creates a layer with:

```
input neurons  = i
output neurons = o
```

Internal matrices:

```
W : o × i
b : o × 1
```

---

## Activation functions

Two function objects must be assigned:

```
activation
activation_d
```

Where:

```
activation   activation function
activation_d derivative of activation
```

Both accept:

```
function pointers
function objects
std::function
```

---

## init

```
void init(T low = -1, T high = 1)
```

Randomly initializes:

```
W
b
```

Values are sampled uniformly from:

```
[low , high]
```

---

## forward

```
Matrix<T> forward(const Matrix<T>& input)
```

Performs forward propagation.

Steps:

```
z = W * input + b
a = activation(z)
```

Returns:

```
a
```

Internal states stored:

```
last_input
z
```

---

## backward

```
Matrix<T> backward(const Matrix<T>& dl_da, const T& step)
```

Performs backpropagation.

Parameters

```
dl_da   derivative of loss with respect to activation output
step    learning rate
```

Important:

```
dl_da is NOT dL/dz
```

It represents:

```
∂L / ∂a
```

The function computes gradients and updates:

```
W
b
```

Returns the gradient for the previous layer.

---

# Activations

Defined in:

```
nn/activations.hpp
```

Activation functions are located inside:

```
LibCN::Activations
```

---

## relu

```
relu(x)
```

```
max(0, x)
```

---

## relu_d

Derivative of ReLU.

---

## leaky_relu

```
leaky_relu(x, alpha)
```

Default

```
alpha = 0.01
```

---

## sigmoid

```
sigmoid(x)
```

```
1 / (1 + exp(-x))
```

---

## sigmoid_d

Derivative of sigmoid.

Implementation uses:

```
sigmoid(x) * (1 - sigmoid(x))
```

This recomputation may introduce **performance overhead**.

---

## tanh

Hyperbolic tangent.

---

## identity

```
identity(x)
```

Returns input directly.

Useful for output layers.

---

# Network

Defined in:

```
nn/network.hpp
```

## Overview

`Network<T>` represents a neural network composed of multiple layers.

Internal storage:

```
std::vector<Layer<T>> layers
```

---

## Constructors

### Default constructor

```
Network()
```

Creates an empty network.

---

### Network(layer_size, in_size, out_size, step)

```
Network(size_t layer_size,
        size_t in_size,
        size_t out_size,
        const T& step)
```

Parameters

```
layer_size   number of layers
in_size      network input size
out_size     network output size
step         learning rate
```

Layers are created but must be configured.

---

## setLayer

```
void setLayer(size_t index, size_t i, size_t o)
```

Sets the size of a specific layer.

```
i = input neurons
o = output neurons
```

---

## setLayerFun

```
void setLayerFun(size_t index,
                 const std::function<T(T)>& a,
                 const std::function<T(T)>& a_d)
```

Assigns activation function and its derivative.

---

## init

```
void init(T low = -1, T high = 1)
```

Initializes all layers by calling their `init()`.

---

## train

```
void train(const Matrix<T>& input,
           const Matrix<T>& expected)
```

Performs one training step.

Procedure:

1. Forward propagation through all layers
2. Compute loss derivative

```
dl/da = output - expected
```

3. Backpropagation through all layers
4. Update weights and biases

This function is intended to be called inside a training loop.

---

## use

```
Matrix<T> use(const Matrix<T>& input)
```

Runs a forward pass without training.

Used for inference and evaluation.

Returns the output matrix.

---

# Usage Summary

To use the entire library:

```
#include "lib_chest_nn.hpp"
```

This header automatically includes all internal components.

```
nn/matrix.hpp
nn/layer.hpp
nn/activations.hpp
nn/network.hpp
```

No build system or linking is required.