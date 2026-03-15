use crate::activations::ActivationFn;

#[derive(Clone)]
pub struct Sigmoid;
impl Sigmoid {
    pub fn new() -> Box<Self> {
        Box::new(Sigmoid)
    }
}

impl ActivationFn for Sigmoid {
    fn apply(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f32, activation: f32) -> f32 {
        activation * (1.0 - activation)
    }
}