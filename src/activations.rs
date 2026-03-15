pub mod sigmoid;
pub mod leaky_relu;

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