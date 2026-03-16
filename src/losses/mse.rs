use nalgebra::{DVector, DVectorView};
use crate::losses::{LossFn, LossFnError, check_sizes};

pub struct MSE;
impl LossFn for MSE {
    fn apply(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<f32, LossFnError> {
        check_sizes(output.len(), expected_output.len())?;

        Ok(output
            .iter()
            .zip(expected_output.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>() / output.len() as f32)
    }

    fn partial_gradient(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<DVector<f32>, LossFnError> {
        check_sizes(output.len(), expected_output.len())?;

        Ok(DVector::from_vec(output
            .iter()
            .zip(expected_output.iter())
            .map(|(x, y)| 2.0 * (x - y) / output.len() as f32)
            .collect()))
    }
}