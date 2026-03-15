# Lib Chest NeuralNetwork (LibCN)

A lightweight, header-only neural network library written in modern C++.

LibCN is a small C++ library built for learning, experimentation, and simple native projects. In **v3.0.0**, the library replaces the old `Matrix<T>` core container with a more general **`Tensor<T>`** type and renames the high-level network type from `Network` to **`MLP`**.

The project focuses on clarity, directness, and hackability rather than framework-scale abstraction.

---

## Highlights of v3.0.0

Compared with the previous version, **LibCN v3.0.0** mainly introduces:

- `Tensor<T>` fully replaces `Matrix<T>` as the core data container
- high-level network type renamed from `Network<T>` to `MLP<T>`
- fully connected layers still use 2D tensors internally for weights, bias, input, and output
- save/load interfaces added for layer parameters:
  - `saveLayerWeights(...)`
  - `saveLayerBias(...)`
  - `loadLayerWeights(...)`
  - `loadLayerBias(...)`
- tensor-based initialization helpers added for matrix-style data construction

Although the public container is now generalized, the current MLP implementation still expects **column-vector shaped 2D tensors** as layer inputs and outputs.

---

## Design Goals

LibCN is designed to:

- help understand how neural networks work internally
- stay small enough to read directly from source
- avoid external dependencies
- remain easy to integrate into normal C++ projects
- provide a lightweight base for experimentation

This project is **not intended to replace PyTorch, TensorFlow, or other industrial frameworks**.

---

## Features

- Header-only library
- Pure template implementation
- Requires only the C++ standard library
- No external dependencies
- General-purpose `Tensor<T>` container
- Fully connected neural network (`MLP`)
- Multiple activation functions
- Multiple loss functions
- Optional specialized path for `softmax + cross_entropy`
- Optional loss printing during training
- Layer weight/bias export and import

---

## Requirements

- **C++20** or newer
- Any modern compiler such as:
  - GCC
  - Clang
  - MSVC

Example compilation:

```bash
g++ -std=c++20 example.cpp -o example
```

---

## Installation

LibCN is header-only.

Copy the headers into your project and include:

```cpp
#include "lib_chest_nn.hpp"
```

No build system, package manager, or separate linking step is required.

---

## Quick Example

A simple XOR example using the current v3.0.0 API:

```cpp
#include "lib_chest_nn.hpp"
#include <iostream>

using namespace std;
using namespace LibCN;

int main(){
    cout<<"this is an example for demostrating the train and use of xor using LibCN"<<endl;
    MLP<float>net(2,2,1,0.05f);

    net.setLoss(Losses::MSE<float>,Losses::MSE_d<float>);

    net.setLayer(0,2,4);
    net.setLayer(1,4,1);

    net.init(-0.5f,0.5f);

    net.setLayerFun(0,Activations::tanh<float>,Activations::tanh_d<float>);
    net.setLayerFun(1,Activations::sigmoid<float>,Activations::sigmoid_d<float>);

    cout<<"MLP network initialized"<<endl;

    Tensor<float>x1=Tensor<float>::matrix({
        {0},
        {0}
    });
    Tensor<float>x2=Tensor<float>::matrix({
        {0},
        {1}
    });
    Tensor<float>x3=Tensor<float>::matrix({
        {1},
        {0}
    });
    Tensor<float>x4=Tensor<float>::matrix({
        {1},
        {1}
    });

    Tensor<float>y1=Tensor<float>::matrix({{0}});
    Tensor<float>y2=Tensor<float>::matrix({{1}});
    Tensor<float>y3=Tensor<float>::matrix({{1}});
    Tensor<float>y4=Tensor<float>::matrix({{0}});   

    cout<<"training data prepared"<<endl;

    cout<<"before training"<<endl;
    cout<<"0 xor 0 -> "<<net.use(x1)<<endl;
    cout<<"0 xor 1 -> "<<net.use(x2)<<endl;
    cout<<"1 xor 0 -> "<<net.use(x3)<<endl;
    cout<<"1 xor 1 -> "<<net.use(x4)<<endl;

    for(int i=0;i<50000;++i){
        if(i%5000==0){
            net.train_p(x1,y1);
            net.train_p(x2,y2);
            net.train_p(x3,y3);
            net.train_p(x4,y4);
        }
        else{
            net.train(x1, y1);
            net.train(x2, y2);
            net.train(x3, y3);
            net.train(x4, y4);
        }
    }

    cout<<"\nafter training"<<endl;
    cout<<"0 xor 0 -> "<<net.use(x1)<<endl;
    cout<<"0 xor 1 -> "<<net.use(x2)<<endl;
    cout<<"1 xor 0 -> "<<net.use(x3)<<endl;
    cout<<"1 xor 1 -> "<<net.use(x4)<<endl;

    auto w0=net.saveLayerWeights(0);
    auto b0=net.saveLayerBias(0);
    auto w1=net.saveLayerWeights(1);
    auto b1=net.saveLayerBias(1);

    cout<<"theta saved"<<endl;

    MLP<float>test_net(2,2,1,0);

    test_net.setLayer(0,2,4);
    test_net.setLayer(1,4,1);

    test_net.setLayerFun(0,Activations::tanh<float>,Activations::tanh_d<float>);
    test_net.setLayerFun(1,Activations::sigmoid<float>,Activations::sigmoid_d<float>);

    cout<<"new MLP network created"<<endl;

    test_net.loadLayerWeights(0,w0);
    test_net.loadLayerBias(0,b0);
    test_net.loadLayerWeights(1,w1);
    test_net.loadLayerBias(1,b1);

    cout<<"theta loaded"<<endl;

    while(true){
        cout<<"please input two booleans (1 or 0), or input other value to quit"<<endl;
        Tensor<float>x(2,{2,1});
        if(!(cin>>x(0,0)>>x(1,0)))break;
        auto yt=test_net.use(x);
        cout<<(yt(0,0)>0.5?"true" : "false")<<endl;
    }

    return 0;
}

```

---

## Tensor Notes

`Tensor<T>` is now the base numerical type of the library.

Currently implemented features include:

- scalar / vector / matrix / higher-dimensional construction
- shape and stride storage
- coordinate mapping
- element access through `operator()`
- transpose by two axes
- element-wise operations
- full accumulation
- axis-wise sum
- Hadamard product
- scalar multiplication
- 2D matrix multiplication through `matrixMultiplication(...)`
- dimension promotion through `ascend()`

For current MLP usage, the most important convention is:

- **inputs and outputs should be 2D tensors shaped like column vectors**
- for example, a 2-input sample should usually be shaped as `{2, 1}`

---

## Activation Functions

Current activation functions are provided in `LibCN::Activations`:

- `relu`
- `relu_d`
- `leaky_relu`
- `leaky_relu_d`
- `sigmoid`
- `sigmoid_d`
- `tanh`
- `tanh_d`
- `identity`
- `identity_d`
- `softmax`
- `softmax_d`

All current activation functions use the form:

```cpp
Tensor<T> f(const Tensor<T>&)
```

`softmax_d` is currently an **approximate element-wise form**, not the full Jacobian.

---

## Loss Functions

Current loss functions are provided in `LibCN::Losses`:

- `MSE`
- `MSE_d`
- `MAE`
- `MAE_d`
- `cross_entropy`
- `cross_entropy_d`

Loss selection is done through:

```cpp
net.setLoss(loss_function, loss_derivative_function);
```

---

## Specialized Softmax + Cross Entropy Path

LibCN contains a specialized training path for the common combination:

- output layer uses `softmax`
- loss uses `cross_entropy`

This path is controlled by two flags already present in the library:

```cpp
net.ce = true;
net.layers.back().sm = true;
```

When both conditions are true, the last layer uses:

```cpp
output - expected
```

as `dL/dz`, and backpropagates through `backward_dz(...)`.

---

## Project Structure

```text
lib_chest_nn.hpp
nn/
    tensor.hpp
    layer.hpp
    activations.hpp
    losses.hpp
    network.hpp
```

### File Overview

**lib_chest_nn.hpp**  
Main entry header.

**nn/tensor.hpp**  
General-purpose tensor container and basic tensor operations.

**nn/layer.hpp**  
Fully connected layer implementation.

**nn/activations.hpp**  
Activation functions and their derivatives.

**nn/losses.hpp**  
Loss functions and their derivatives.

**nn/network.hpp**  
`MLP<T>` definition and training logic.

---

## Current Scope

LibCN v3.0.0 is currently suitable for:

- learning neural networks
- educational demonstrations
- small experiments
- lightweight native C++ usage
- testing tensor-based neural-network code without external frameworks

It is **not intended for production-scale training workloads**.

---

## Author

MrChest / 石函

---

## License

This project is released under the MIT License.

See the `LICENSE` file for details.
