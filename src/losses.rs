mod mse;
pub use mse::MSE;

mod bce;
pub use bce::BCE;

use nalgebra::{DVector, DVectorView};
use thiserror::Error;

pub trait LossFn {
    fn apply(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<f32, LossFnError>;

    fn partial_gradient(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<DVector<f32>, LossFnError>;
}

#[derive(Debug, Error)]
pub enum LossFnError {
    #[error("given output size ({given_output_size}) does not equal expected output size ({expected_output_size})")]
    OutputSizeMismatch {
        given_output_size: usize,
        expected_output_size: usize,
    },
}

fn check_sizes(output_size: usize, expected_output_size: usize) -> Result<(), LossFnError> {
    if output_size != expected_output_size {
        return Err(LossFnError::OutputSizeMismatch {
            given_output_size: output_size,
            expected_output_size: expected_output_size,
        });
    }

    Ok(())
}