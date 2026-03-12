# Lib Chest NeuralNetwork (LibCN)

A lightweight neural network library written in pure C++.

Lib Chest NeuralNetwork (LibCN) is a small, header-only C++ library designed for learning and experimenting with neural networks and machine learning algorithms. It focuses on simplicity, transparency, and ease of use within native C++ environments.

This project is primarily intended for educational purposes, experimentation, and as a reference implementation.

---

## Motivation

Many powerful machine learning frameworks already exist, such as PyTorch and TensorFlow.  
These frameworks are excellent and highly optimized.

However, their C++ interfaces are often extremely complex and difficult to use for everyday development or learning purposes.

LibCN attempts to provide an alternative approach:

- A **pure C++ neural network implementation**
- Minimal abstraction
- Easy to read and modify
- Designed to feel natural inside normal C++ code

This library is **not intended to replace large frameworks**, but rather to provide a simple and understandable implementation that can be used for learning, experimentation, and reference.

---

## Goals

LibCN is designed with the following goals:

- Learn how neural networks work internally
- Implement machine learning algorithms from scratch
- Provide a simple matrix utility
- Avoid heavy dependencies
- Be easy to integrate into existing C++ projects

---

## Features

- Header-only library
- Pure template implementation
- Requires only the C++ standard library
- No external dependencies
- No build system required
- Works with any C++20 compatible compiler
- Simple matrix operations
- Basic neural network components
- Activation functions
- Layer abstraction
- Easy integration into C++ projects

---

## Design Philosophy

LibCN follows a few simple principles:

1. **Transparency over abstraction**

   Code should be easy to read and understand.

2. **Minimal dependencies**

   Only the C++ standard library is used.

3. **Header-only simplicity**

   No build systems, no linking steps.

4. **C++ friendly interface**

   Designed to feel natural for C++ developers.

---

## Requirements

- C++20 or newer
- Any modern C++ compiler

Examples:

- GCC
- Clang
- MSVC

Example compilation:

```
g++ -std=c++20 example.cpp
```

---

## Installation

LibCN is a header-only library.

Simply copy the repository into your project and include the main header:

```
#include "lib_chest_nn.hpp"
```

No build system, installation script, or package manager is required.

---

## Quick Example

A minimal example using LibCN:

```cpp

#include "lib_chest_nn.hpp"
#include <iostream>

using namespace std;
using namespace LibCN;

int main()
{
    Network<float> net(2, 2, 1, 0.05f);

    net.setLayer(0, 2, 4);
    net.setLayer(1, 4, 1);

    net.init(-0.5f, 0.5f);

    net.setLayerFun(0, Activations::tanh<float>, Activations::tanh_d<float>);
    net.setLayerFun(1, Activations::sigmoid<float>, Activations::sigmoid_d<float>);

    Matrix<float> x1{{0},{0}};
    Matrix<float> x2{{0},{1}};
    Matrix<float> x3{{1},{0}};
    Matrix<float> x4{{1},{1}};

    Matrix<float> y1{{0}};
    Matrix<float> y2{{1}};
    Matrix<float> y3{{1}};
    Matrix<float> y4{{0}};

    cout << "before training" << endl;
    cout << "0 xor 0 -> " << net.use(x1) << endl;
    cout << "0 xor 1 -> " << net.use(x2) << endl;
    cout << "1 xor 0 -> " << net.use(x3) << endl;
    cout << "1 xor 1 -> " << net.use(x4) << endl;

    for(int i = 0; i < 50000; ++i)
    {
        net.train(x1, y1);
        net.train(x2, y2);
        net.train(x3, y3);
        net.train(x4, y4);
    }

    cout << "\nafter training" << endl;
    cout << "0 xor 0 -> " << net.use(x1) << endl;
    cout << "0 xor 1 -> " << net.use(x2) << endl;
    cout << "1 xor 0 -> " << net.use(x3) << endl;
    cout << "1 xor 1 -> " << net.use(x4) << endl;

    return 0;
}

```

More examples will be added in the future.

---

## Project Structure

```
lib_chest_nn.hpp
nn/
    matrix.hpp
    layer.hpp
    activations.hpp
    network.hpp
```

### File Overview

**lib_chest_nn.hpp**

Main entry header.  
Including this file provides access to the entire library.

**nn/matrix.hpp**

Matrix implementation and matrix operations.

**nn/layer.hpp**

Definition of neural network layers.

**nn/activations.hpp**

Common activation functions and their derivatives.

**nn/network.hpp**

High level neural network structure.

---

## Current Status

Current version:

```
v1.0.0
```

LibCN is currently suitable for:

- Learning neural networks
- Educational demonstrations
- Small experiments
- Reference implementations

It is **not intended for production-scale deep learning workloads**.

---

## Future Plans

Possible future improvements include:

- More activation functions
- Additional layer types
- Better error diagnostics
- Training utilities
- Optimization algorithms
- Optional GPU support

---

## Author

MrChest / 石函

---

## License

This project is released under the MIT License.

See the `LICENSE` file for details.