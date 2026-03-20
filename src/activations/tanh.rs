use crate::activations::ActivationFn;
use std::any::Any;

#[derive(Clone)]
pub struct Tanh;
impl Tanh {
    pub fn new() -> Box<Self> {
        Box::new(Tanh)
    }
}

impl ActivationFn for Tanh {
    fn apply(&self, x: f32) -> f32 {
        let e = (-2.0 * x.abs()).exp();
        ((1.0 - e) / (1.0 + e)).copysign(x)
    }

    fn derivative(&self, x: f32, activation: f32) -> f32 {
        1.0 - activation * activation
    }

    fn as_any(&self) -> &dyn Any { self }
}