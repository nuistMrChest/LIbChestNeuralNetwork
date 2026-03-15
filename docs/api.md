# LibCN API Documentation

This document describes the public interfaces of **Lib Chest NeuralNetwork (LibCN)** in the current **v3.0.0** code.

To use the whole library, include:

```cpp
#include "lib_chest_nn.hpp"
```

The entry header includes:

```cpp
./nn/layer.hpp
./nn/activations.hpp
./nn/network.hpp
./nn/losses.hpp
./nn/tensor.hpp
```

---

# Tensor

Defined in:

```cpp
nn/tensor.hpp
```

## Overview

`Tensor<T>` is the core mathematical container used throughout LibCN v3.0.0.

It stores:

- tensor dimension count
- shape
- stride
- flattened values in contiguous storage

The current implementation supports scalar tensors, vectors, matrices, and several higher-dimensional initializer-list constructors.

---

## Template Requirement

`Tensor<T>` requires `T` to satisfy the `Element` concept.

The current concept checks support for:

```cpp
+  +=
-  -=
*  *=
/
> < >= <= == !=
std::ostream << value
```

---

## Data Members

```cpp
size_t dimension;
std::vector<size_t> shape;
std::vector<size_t> stride;
std::vector<T> values;
```

---

## Basic Member Functions

### getDimention

```cpp
size_t getDimention() const
```

Returns the tensor dimension count.

> Note: the function name in the current code is spelled `getDimention()`.

---

### getShape

```cpp
const std::vector<size_t>& getShape() const
```

Returns the stored shape vector.

---

### unravel_index

```cpp
std::vector<size_t> unravel_index(size_t index) const
```

Converts a flat index into multi-dimensional coordinates according to the tensor stride.

---

### ravel_index

```cpp
size_t ravel_index(const std::vector<size_t>& idx) const
```

Converts multi-dimensional coordinates into a flat index.

---

### setStride

```cpp
void setStride()
```

Recomputes the internal stride array from the current shape and dimension.

---

### resize

```cpp
void resize(size_t d, const std::vector<size_t>& s)
```

Resets the tensor to dimension `d`, shape `s`, resizes storage, and rebuilds stride.

---

## Constructors

### Default constructor

```cpp
Tensor()
```

Creates an empty tensor:

- `dimension = 0`
- `shape` empty
- `values` empty

---

### General shape constructor

```cpp
Tensor(size_t d, const std::vector<size_t>& s)
```

Creates a tensor with dimension `d`, shape `s`, and storage size equal to the product of all extents.

---

### Copy constructor

```cpp
Tensor(const Tensor<T>& a)
```

Creates a deep copy.

---

### Scalar constructor

```cpp
Tensor(const T& a)
```

Creates a 0-dimensional tensor containing one value.

---

### Vector constructor

```cpp
Tensor(const std::vector<T>& a)
```

Creates a 1-dimensional tensor from a vector.

---

### 1D initializer-list constructor

```cpp
Tensor(std::initializer_list<T> a)
```

Creates a 1D tensor.

Example:

```cpp
Tensor<int> a{1, 2, 3};
```

---

### 2D initializer-list constructor

```cpp
Tensor(std::initializer_list<std::initializer_list<T>> a)
```

Creates a 2D tensor.

Example:

```cpp
Tensor<float> a{{1, 2}, {3, 4}};
```

---

### 3D initializer-list constructor

```cpp
Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> a)
```

Creates a 3D tensor.

---

### 4D initializer-list constructor

```cpp
Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> a)
```

Creates a 4D tensor.

---

### matrix helper

```cpp
static Tensor<T> matrix(std::initializer_list<std::initializer_list<T>> a)
```

Convenience helper for creating a 2D tensor explicitly in matrix form.

This is especially useful for MLP inputs and outputs, where column-vector shaped 2D tensors are expected.

Example:

```cpp
auto x = Tensor<float>::matrix({
    {1},
    {0}
});
```

---

## Element Access

### Mutable access

```cpp
template<typename... Args>
T& operator()(Args... args)
```

### Const access

```cpp
template<typename... Args>
const T& operator()(Args... args) const
```

Accesses an element by coordinates.

Example:

```cpp
a(1, 0)
```

The current implementation trusts the caller and does not perform bounds checking.

---

## Output

```cpp
friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& a)
```

Printing behavior depends on the dimension:

- empty tensor → `{ NULL }`
- 0D tensor → scalar form
- 1D tensor → flat braced list
- 2D tensor → row/column style output
- higher dimension → recursive formatted output

---

## Arithmetic Operations

### Addition

```cpp
Tensor<T> operator+(const Tensor<T>& a) const
Tensor<T>& operator+=(const Tensor<T>& a)
```

Performs element-wise addition when dimension and shape match.

If shapes do not match:

- `operator+` returns an empty tensor
- `operator+=` leaves the object unchanged

---

### Subtraction

```cpp
Tensor<T> operator-(const Tensor<T>& a) const
Tensor<T>& operator-=(const Tensor<T>& a)
```

Performs element-wise subtraction when dimension and shape match.

---

### Hadamard product

```cpp
Tensor<T> hadamard(const Tensor<T>& a) const
Tensor<T>& hadamard_self(const Tensor<T>& a)
```

Performs element-wise multiplication when shape matches.

---

### Scalar multiplication

```cpp
Tensor<T> operator*(const T& a) const
Tensor<T>& operator*=(const T& a)
friend Tensor<T> operator*(const T& a, const Tensor<T>& b)
```

Multiplies every element by a scalar.

---

### Tensor multiplication operator

```cpp
Tensor<T> operator*(const Tensor<T>& a) const
Tensor<T>& operator*=(const Tensor<T>& a)
```

In the current code, this operator is **not general tensor multiplication**.

Current behavior:

- if `a.dimension == 0`, it behaves like scalar multiplication by `a.values[0]`
- if `this->dimension == 0`, it behaves like scalar multiplication by `this->values[0]`
- otherwise it returns an empty tensor

For 2D matrix multiplication, use `matrixMultiplication(...)` instead.

---

## Shape / Dimension Operations

### transpose

```cpp
Tensor<T> transpose(size_t d1, size_t d2) const
Tensor<T>& transpose_self(size_t d1, size_t d2)
```

Swaps two axes.

If an axis is out of range, `transpose(...)` returns an empty tensor.

---

### sum

```cpp
Tensor<T> sum(size_t axis) const
```

Returns a tensor with one fewer dimension by summing along the given axis.

---

### accumulate

```cpp
Tensor<T> accumulate() const
```

Sums all elements and returns a 0-dimensional tensor.

---

### dot

```cpp
Tensor<T> dot(const Tensor<T>& a) const
```

Computes:

```cpp
hadamard(a).accumulate()
```

So the current `dot(...)` is a full element-wise contraction into a scalar tensor, assuming matching shapes.

---

### matrixMultiplication

```cpp
Tensor<T> matrixMultiplication(const Tensor<T>& b) const
```

Performs ordinary 2D matrix multiplication when:

```cpp
this->dimension == 2
b.dimension == 2
this->shape[1] == b.shape[0]
```

Otherwise returns an empty tensor.

---

### ascend

```cpp
Tensor<T> ascend() const
Tensor<T>& ascend_self()
```

Adds a new leading dimension of size `1`.

For example, a shape `{2, 3}` becomes `{1, 2, 3}`.

---

# MLPLayer

Defined in:

```cpp
nn/layer.hpp
```

## Overview

`MLPLayer<T>` represents one fully connected layer.

Although LibCN now uses `Tensor<T>`, the current MLP layer still works with **2D tensors** for weights, bias, inputs, outputs, and cached states.

---

## Data Members

```cpp
std::function<Tensor<T>(const Tensor<T>&)> activation;
std::function<Tensor<T>(const Tensor<T>&)> activation_d;
size_t in_size;
size_t out_size;
Tensor<T> W;
Tensor<T> b;
Tensor<T> last_input;
Tensor<T> z;
bool sm;
```

### Stored tensor shapes

For `MLPLayer(i, o)`:

- `W` has shape `{o, i}`
- `b` has shape `{o, 1}`
- `last_input` has shape `{i, 1}`
- `z` has shape `{o, 1}`

### sm

`sm` defaults to `false`.

It is used as a marker for the specialized `softmax + cross_entropy` path.

---

## Constructors

### Default constructor

```cpp
MLPLayer()
```

Creates an empty layer.

### Sized constructor

```cpp
MLPLayer(size_t i, size_t o)
```

Creates a fully connected layer with input size `i` and output size `o`.

---

## init

```cpp
void init(T low = T(-1), T high = T(1))
```

Initializes `W` and `b` using a uniform real distribution on `[low, high]`.

The current implementation uses:

```cpp
static std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<T> dist(low, high);
```

---

## forward

```cpp
Tensor<T> forward(const Tensor<T>& input)
```

Performs:

```cpp
z = W.matrixMultiplication(input) + b
activation(z)
```

Expected input shape is `{in_size, 1}`.

If the shape check fails, `z` is not updated, and the function still returns `activation(z)` using the previously stored `z`.

---

## backward

```cpp
Tensor<T> backward(const Tensor<T>& dl_da, const T& step)
```

Standard backpropagation path.

Computation:

```cpp
dl_dz = dl_da.hadamard(activation_d(z))
res   = W.transpose(0,1).matrixMultiplication(dl_dz)
W    -= step * (dl_dz.matrixMultiplication(last_input.transpose(0,1)))
b    -= step * dl_dz
```

Returns gradient with respect to the previous layer output.

---

## backward_dz

```cpp
Tensor<T> backward_dz(const Tensor<T>& dl_dz, const T& step)
```

Backpropagation path used when the caller already has `dL/dz`.

This is used by the specialized `softmax + cross_entropy` path in `MLP<T>`.

---

# Activations

Defined in:

```cpp
nn/activations.hpp
```

All activation functions are in:

```cpp
LibCN::Activations
```

All current activation functions use the form:

```cpp
Tensor<T> f(const Tensor<T>& a)
```

## Available activations

```cpp
relu
relu_d
leaky_relu
leaky_relu_d
sigmoid
sigmoid_d
tanh
tanh_d
identity
identity_d
softmax
softmax_d
```

---

## Notes on current implementation

The activation implementations currently iterate with two nested loops over:

```cpp
a.getShape()[0]
a.getShape()[1]
```

So while the function signatures accept `Tensor<T>`, the current activation code is effectively written for **2D tensors**.

### softmax

The implementation uses max-subtraction for basic numerical stabilization:

```cpp
exp(a(i,j) - mx)
```

and normalizes by the sum of all exponentials.

### softmax_d

The current implementation computes element-wise:

```cpp
s(i,j) * (1 - s(i,j))
```

This is an approximation and **not the full softmax Jacobian**.

---

# Losses

Defined in:

```cpp
nn/losses.hpp
```

All loss functions are in:

```cpp
LibCN::Losses
```

## Available losses

```cpp
MSE
MSE_d
MAE
MAE_d
cross_entropy
cross_entropy_d
```

All current loss implementations also operate with two nested loops over the first two dimensions, so they are effectively written for **2D tensors** in the present code.

---

## MSE

```cpp
template<Element T>
T MSE(const Tensor<T>& x, const Tensor<T>& e)
```

If shapes match, computes:

```cpp
sum((x - e)^2) / 2
```

---

## MSE_d

```cpp
template<Element T>
Tensor<T> MSE_d(const Tensor<T>& x, const Tensor<T>& e)
```

Returns:

```cpp
x - e
```

---

## MAE

```cpp
template<Element T>
T MAE(const Tensor<T>& x, const Tensor<T>& e)
```

If shapes match, computes mean absolute error.

> In the current source, the final division is written as `sum / (x.h * x.l)`. Since `Tensor<T>` does not define `h` and `l`, this appears inconsistent with the rest of the v3.0.0 tensor-based code and should likely be corrected to a shape-based expression.

---

## MAE_d

```cpp
template<Element T>
Tensor<T> MAE_d(const Tensor<T>& x, const Tensor<T>& e)
```

For matching shapes, returns element-wise:

- `1` when `x(i,j) >= e(i,j)`
- `-1` otherwise

---

## cross_entropy

```cpp
template<Element T>
T cross_entropy(const Tensor<T>& x, const Tensor<T>& e)
```

For matching shapes, computes:

```cpp
- sum(e(i,j) * log(v))
```

where `v` is clamped to:

```cpp
[1e-12, 1 - 1e-12]
```

---

## cross_entropy_d

```cpp
template<Element T>
Tensor<T> cross_entropy_d(const Tensor<T>& x, const Tensor<T>& e)
```

Returns element-wise:

```cpp
- e(i,j) / v
```

with the same clamp behavior.

If shapes do not match, the current code returns a tensor with the same dimension and shape as `x`, because `res` is constructed before the mismatch check.

---

# MLP

Defined in:

```cpp
nn/network.hpp
```

## Overview

`MLP<T>` is the high-level feed-forward neural network type in LibCN v3.0.0.

This is the renamed successor to the old `Network<T>` interface.

---

## Data Members

```cpp
size_t in_size;
size_t out_size;
std::vector<MLPLayer<T>> layers;
T step;
std::function<T(const Tensor<T>&, const Tensor<T>&)> loss;
std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)> loss_d;
bool ce;
```

### ce

`ce` defaults to `false`.

It is used as a flag for the specialized `softmax + cross_entropy` training path.

---

## Constructors

### Default constructor

```cpp
MLP()
```

Creates an empty network.

### Sized constructor

```cpp
MLP(size_t layer_size, size_t in_size, size_t out_size, const T& step)
```

Initializes:

- number of layers
- input size
- output size
- learning rate

The layers are default-constructed and must be configured afterward.

---

## Parameter Save / Load

### saveLayerWeights

```cpp
Tensor<T> saveLayerWeights(size_t index)
```

Returns a copy of `layers[index].W`.

### saveLayerBias

```cpp
Tensor<T> saveLayerBias(size_t index)
```

Returns a copy of `layers[index].b`.

### loadLayerWeights

```cpp
void loadLayerWeights(size_t index, const Tensor<T>& weights)
```

Replaces `layers[index].W`.

### loadLayerBias

```cpp
void loadLayerBias(size_t index, const Tensor<T>& bias)
```

Replaces `layers[index].b`.

These interfaces expose raw tensor parameters and leave external serialization format decisions to the user.

---

## Configuration

### setLayer

```cpp
void setLayer(size_t index, size_t i, size_t o)
```

Sets `layers[index]` to `MLPLayer<T>(i, o)`.

### setLayerFun

```cpp
void setLayerFun(
    size_t index,
    const std::function<Tensor<T>(const Tensor<T>&)>& a,
    const std::function<Tensor<T>(const Tensor<T>&)>& a_d)
```

Assigns activation function and derivative for one layer.

### setLoss

```cpp
void setLoss(
    const std::function<T(const Tensor<T>&, const Tensor<T>&)> l,
    const std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)> l_d)
```

Assigns loss function and loss derivative.

### init

```cpp
void init(T low = T(-1), T high = T(1))
```

Calls `init(low, high)` for every layer.

---

## Inference

### use

```cpp
Tensor<T> use(const Tensor<T>& input)
```

Feeds the input through all layers and returns the final output.

The intended input format is a 2D tensor shaped like a column vector.

---

## Training

### train

```cpp
void train(const Tensor<T>& input, const Tensor<T>& expected)
```

Performs one forward pass and one backward pass.

#### Ordinary path

Used when:

```cpp
!(layers.back().sm && ce)
```

Then:

```cpp
last_dl_da = loss_d(output, expected)
```

and layers are backpropagated from last to first with `backward(...)`.

#### Specialized softmax + cross_entropy path

Used when:

```cpp
layers.back().sm && ce
```

Then the last layer uses:

```cpp
dl_dz = output - expected
```

and backpropagates through:

```cpp
layers.back().backward_dz(dl_dz, step)
```

Previous layers still use ordinary `backward(...)`.

---

### train_p

```cpp
void train_p(const Tensor<T>& input, const Tensor<T>& expected)
```

Same as `train(...)`, but prints the current loss first:

```cpp
std::cout << "Loss: " << loss(output, expected) << std::endl;
```

---

# Typical Usage Pattern

```cpp
MLP<float> net(layer_count, input_size, output_size, learning_rate);

net.setLoss(...);
net.setLayer(0, ...);
net.setLayer(1, ...);
net.setLayerFun(0, ..., ...);
net.setLayerFun(1, ..., ...);
net.init();
```

Input / output tensors should generally be prepared as 2D column vectors, for example:

```cpp
auto x = Tensor<float>::matrix({
    {1},
    {0}
});
```

---

# Summary of Renamed v3.0.0 Interfaces

Compared with the previous API:

```cpp
Network<T>      -> MLP<T>
setMLPLayer     -> setLayer
setMLPLayerFun  -> setLayerFun
Matrix<T>       -> Tensor<T>
```

---

# Notes

This documentation is written against the uploaded v3.0.0 source itself, so it reflects the current code behavior, including implementation details and current limitations.
