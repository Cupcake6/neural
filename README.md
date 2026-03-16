# neural
A small machine learning library in Rust that allows training and executing simple feedforward neural networks

## Motivation
I built this project to get a grasp of how neural networks and machine learning work internally. While implementing it, I learned the basics of multivariable calculus behind gradient descent and backpropagation.

## Features
- Fully customizable multilayer feedforward networks
- Arbitrary layer sizes and per-layer activation functions
- Support for custom activation and loss functions
- Gradient computation via backpropagation

## Getting started
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

## Example usage
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
    &[2, 4, 4, 1], 
    &[Sigmoid::new(), LeakyReLU::new(0.5), Swish::new()],
    &distr::Uniform::new(-0.5, 0.5).unwrap()
).unwrap();

// Perform one iteration of training on a dataset
network.learn(dataset, &MSE, 0.01).unwrap();

// Feed inputs into the network
let outputs = network.forward(dvector![
    0.0, 1.0
]).unwrap();
```

## Architecture
- A network is a sequence of layers
- Each layer stores its weights, biases and activation function
- The network trains using gradient descent
- The gradient is computed via backpropagation
- Activation and loss functions are regular Rust structs
- `nalgebra` is used for matrix/vector operations

## Future improvements
- Model serialization / saving and loading
- Additional initialization methods
- Mini-batch training support
- Remove `nalgebra` from the public interface