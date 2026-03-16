mod sigmoid;
pub use sigmoid::Sigmoid;

mod leaky_relu;
pub use leaky_relu::LeakyReLU;

mod tanh;
pub use tanh::Tanh;

mod swish;
pub use swish::Swish;

pub trait ActivationFn: 'static + ActivationFnClone {
    fn apply(&self, x: f32) -> f32;
    fn derivative(&self, x: f32, activation: f32) -> f32;
}

pub trait ActivationFnClone {
    fn clone_box(&self) -> Box<dyn ActivationFn>;
}

impl<T> ActivationFnClone for T
where
    T: 'static + ActivationFn + Clone,
{
    fn clone_box(&self) -> Box<dyn ActivationFn> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ActivationFn> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}