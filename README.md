# neural
A small machine learning library in Rust that allows training and executing simple feedforward neural networks

## Motivation
I built this project to get a grasp of how neural networks and machine learning work internally. While implementing it, I learned the basics of multivariable calculus behind gradient descent and backpropagation.

## Features
- Multilayer feedforward architecture
- Arbitrary layer sizes
- Custom activation function
- Custom loss functions
- Gradient computation via backpropagation
- Different activation functions per layer

## Example usage
``` rust
// Create a new network with random weights and biases
let mut network = Network::random(
    &[2, 4, 4, 1], 
    &[Sigmoid::new(), LeakyReLU::new(0.5), Swish::new()],
    &Uniform::new(-0.5, 0.5).unwrap()
).unwrap();

// Perform one iteration of training on a dataset
network.learn(&dataset, &losses::MSE, 0.01).unwrap();

// Feed inputs into the network
let outputs = network.forward(dvector![
  1.0, 2.0
]).unwrap()
```

## Architecture
- A network is a sequence of layers
- Each layer stores its weights, biases and activation function
- The network trains using gradient descent
- The gradient is computed via backpropagation
- Activation and loss functions are regular Rust structs
- `nalgebra` is used for matrix/vector operations

## Future improvements
- Stop using `nalgebra` in the library interface
- Model serialization
- Better documentation
- Mini-batch training
- More model initializers