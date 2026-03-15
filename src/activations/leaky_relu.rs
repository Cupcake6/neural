use crate::activations::ActivationFn;

#[derive(Clone)]
pub struct LeakyReLU(f32);
impl LeakyReLU {
    pub fn new(slope: f32) -> Box<Self> {
        Box::new(LeakyReLU(slope))
    }
}

impl ActivationFn for LeakyReLU {
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 { x }
        else { x * self.0 }
    }

    fn derivative(&self, x: f32, activation: f32) -> f32 {
        if x > 0.0 { 1.0 }
        else { self.0 }
    }
}