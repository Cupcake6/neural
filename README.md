# neural
A small machine learning library in Rust that allows training and executing simple feedforward neural networks.

## Check out [some examples of it being in action](https://github.com/Cupcake6/neural-examples).

# Motivation
I built this project to understand how neural networks work internally by _implementing them from scratch_. I learned how the most basic feedforward networks are structured, how backpropagation computes the gradient of the loss function and the calculus behind it. I was able to turn those concepts into a working-yet-minimal library.

# Features
- Fully customizable multilayer feedforward networks
- Arbitrary layer sizes and per-layer activation functions
- Support for custom activation and loss functions
- Gradient computation via backpropagation
- Model serialization / saving and loading

# Getting started
Clone the repository or add it as a submodule:
``` shell
git clone https://github.com/Cupcake6/neural.git
```

In your `Cargo.toml` add:
``` toml
[dependencies]
neural = { path = "path/to/cloned/repository" }

# Also add nalgebra for interfacing with the library
nalgebra = "0.34.1"
```

# Example usage
``` rust
use neural::prelude::*;
use nalgebra::dvector;

let dataset = &[ // XOR function training dataset
    Sample::new(dvector![0.0, 0.0], dvector![0.0]),
    Sample::new(dvector![1.0, 0.0], dvector![1.0]),
    Sample::new(dvector![0.0, 1.0], dvector![1.0]),
    Sample::new(dvector![1.0, 1.0], dvector![0.0]),
];

// Create a new network with random weights and biases
let mut network = Network::random(
    &[2, 2, 1], 
    &[Swish::new(), Sigmoid::new()],
    &distr::Uniform::new(-0.5, 0.5).unwrap()
).unwrap();

// Perform 2000 iterations of training on the dataset
for _ in 0..2000 {
    network.learn(dataset, &BCE, 0.4).unwrap();
}

// The model now approximates the XOR function reasonably well

// Feed inputs into the network
let output = network.forward(dvector![
    0.0, 1.0
]).unwrap();

// output[0] ~= 1.0
```

# Architecture
- A network is a sequence of layers
- Each layer stores its weights, biases and activation function
- The network trains using gradient descent
- The gradient is computed via backpropagation
- Activation and loss functions are regular Rust structs
- `nalgebra` is used for matrix/vector operations

# What I learned
- Basic calculus, including derivatives
- How those concepts extend into multivariable calculus
- How neural networks are structured
- Why gradient descent works this wonderfully
- How backpropagation works mathematically and in code

# Limitations
- Runs on a single CPU core
- Only supports fully connected feedforward networks
- No automatic differentiation

# Future improvements
- Additional initialization methods
- Mini-batch training support
- Remove `nalgebra` from the public interface