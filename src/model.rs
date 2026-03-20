use std::{io::Write, path::Path};
use std::fs::File;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use crate::{
    activations::{
        LeakyReLU,
        Sigmoid,
        Swish,
        Tanh,
    }, network::{self, Network}, prelude::ActivationFn
};

mod model_v1;

#[derive(Serialize, Deserialize)]
pub enum ModelFile {
    V1(model_v1::Model),
}

#[derive(Debug, Error)]
pub enum SerializationError {
    #[error("activation function on layer {0} is not supported")]
    ActivationFunctionNotSupported(usize),
}

impl ModelFile {
    fn get_activation_function(layer: &network::layer::Layer) -> Option<model_v1::ActivationFn> {
        let activation_fn = layer.get_activation_fn();
        let trait_obj: &dyn ActivationFn = &**activation_fn;

        if let Some(a) = trait_obj.as_any().downcast_ref::<LeakyReLU>() {
            return Some(model_v1::ActivationFn::LeakyReLU(a.0))
        }

        if let Some(_) = trait_obj.as_any().downcast_ref::<Sigmoid>() {
            return Some(model_v1::ActivationFn::Sigmoid)
        }

        if let Some(_) = trait_obj.as_any().downcast_ref::<Swish>() {
            return Some(model_v1::ActivationFn::Swish)
        }

        if let Some(_) = trait_obj.as_any().downcast_ref::<Tanh>() {
            return Some(model_v1::ActivationFn::Tanh)
        }

        None
    }

    fn make_layer(layer_index: usize, network: &Network) -> Result<model_v1::Layer, SerializationError> {
        assert!(layer_index < network.layers.len());

        let layer = &network.layers[layer_index];

        Ok(model_v1::Layer {
            weights: (0..layer.weight_count())
                .map(|x| *layer.get_weight(x / layer.output_size(), x % layer.output_size()).unwrap())
                .collect(),

            biases: (0..layer.output_size())
                .map(|x| *layer.get_bias(x).unwrap())
                .collect(),

            activation_fn: Self::get_activation_function(layer).ok_or(SerializationError::ActivationFunctionNotSupported(layer_index))?
        })
    }

    pub fn serialize(network: &Network) -> Result<Self, SerializationError> {
        let layers = (0..network.layers.len())
            .map(|x| Self::make_layer(x, network))
            .collect::<Result<Vec<_>, _>>()?;

        let model = model_v1::Model { layers };

        Ok(Self::V1(model))
    }

    pub fn deserialize(&self) -> Network {
        match &self {
            Self::V1(model) => model.deserialize()
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<File> {
        let mut file = File::create(path)?;
        let bytes = postcard::to_stdvec(self)?;
        file.write_all(&bytes)?;

        Ok(file)
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let model_file = postcard::from_bytes(&bytes)?;
        Ok(model_file)
    }
}