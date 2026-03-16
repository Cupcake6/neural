use nalgebra::{DVector, DVectorView};
use crate::losses::{LossFn, LossFnError, check_sizes};

pub struct BCE;
impl BCE {
    const EPSILON: f32 = 0.00001; // To prevent division by 0
}

impl LossFn for BCE {
    fn apply(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<f32, LossFnError> {
        check_sizes(output.len(), expected_output.len())?;

        Ok(output
            .iter()
            .zip(expected_output.iter())
            .map(|(&x, &y)| y * x.ln() + (1.0 - y) * (1.0 - x).ln())
            .sum::<f32>() / -(output.len() as f32))
    }

    fn partial_gradient(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<DVector<f32>, LossFnError> {
        check_sizes(output.len(), expected_output.len())?;

        Ok(DVector::from_vec(output
            .iter()
            .map(|x| x.clamp(BCE::EPSILON, 1.0 - BCE::EPSILON))
            .zip(expected_output.iter())
            .map(|(x, y)| -(y / x - (1.0 - y) / (1.0 - x)))
            .collect()))
    }
}