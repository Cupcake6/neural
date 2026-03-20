use serde::{Serialize, Deserialize};
use crate::{
    activations,
    network::{self, Network}
};

#[derive(Serialize, Deserialize)]
pub enum ActivationFn {
    LeakyReLU(f32),
    Sigmoid,
    Swish,
    Tanh,
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub activation_fn: ActivationFn,
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    fn load_layer(layer: &Layer) -> network::layer::Layer {
        let input_size = layer.weights.len() / layer.biases.len();
        let output_size = layer.biases.len();

        let mut network_layer = network::layer::Layer::zeros(
            input_size,
            output_size,
            match layer.activation_fn {
                ActivationFn::LeakyReLU(x) => activations::LeakyReLU::new(x),
                ActivationFn::Sigmoid => activations::Sigmoid::new(),
                ActivationFn::Swish => activations::Swish::new(),
                ActivationFn::Tanh => activations::Tanh::new(),
            },
        ).unwrap();

        for output_index in 0..output_size {
            for input_index in 0..input_size {
                *network_layer.get_weight_mut(input_index, output_index).unwrap() = layer.weights[output_index + input_index * output_size];
            }
            
            *network_layer.get_bias_mut(output_index).unwrap() = layer.biases[output_index];
        }

        network_layer
    }

    pub fn deserialize(&self) -> Network {
        Network {
            layers: self.layers.iter()
                .map(|layer| Self::load_layer(layer))
                .collect()
        }
    }
}