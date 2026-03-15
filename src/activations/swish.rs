use crate::activations::ActivationFn;

#[derive(Clone)]
pub struct Swish;

impl Swish {
    pub fn new() -> Box<Self> {
        Box::new(Swish)
    }
}

impl ActivationFn for Swish {
    fn apply(&self, x: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        x * sigmoid
    }

    fn derivative(&self, x: f32, _activation: f32) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        sigmoid + x * sigmoid * (1.0 - sigmoid)
    }
}