extern crate rand;
extern crate nalgebra as na;

use na::{DMatrix, DVector};
use rand::Rng;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn main() {
    let inputs = DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let expected_output = DVector::from_column_slice(&[0.0, 1.0, 1.0, 0.0]);

    let epochs = 20000;
    let lr = 0.8;
    let (input_layer_neurons, hidden_layer_neurons, output_layer_neurons) = (2, 2, 1);

    let mut rng = rand::thread_rng();
    let mut hidden_weights = DMatrix::from_fn(input_layer_neurons, hidden_layer_neurons, |_r, _c| rng.gen_range(-0.5..0.5));
    let mut hidden_bias = DMatrix::from_fn(1, hidden_layer_neurons, |_r, _c| rng.gen_range(-0.5..0.5));
    let mut output_weights = DMatrix::from_fn(hidden_layer_neurons, output_layer_neurons, |_r, _c| rng.gen_range(-0.5..0.5));
    let mut output_bias = DMatrix::from_element(1, output_layer_neurons, 0.1);

    for _ in 0..epochs {
        let hidden_layer_activation = &inputs * &hidden_weights + &hidden_bias;
        let hidden_layer_output = hidden_layer_activation.map(|&x| sigmoid(x));

        let output_layer_activation = &hidden_layer_output * &output_weights + &output_bias;
        let predicted_output = output_layer_activation.map(|&x| sigmoid(x));

        // Backpropagation
        let error = &expected_output - &predicted_output;
        let d_predicted_output = &error * &predicted_output.map(|&x| sigmoid_derivative(x));

        let error_hidden_layer = d_predicted_output * output_weights.transpose();
        let d_hidden_layer = &error_hidden_layer * &hidden_layer_output.map(|&x| sigmoid_derivative(x));

        // Updating Weights and Biases
        output_weights += hidden_layer_output.transpose() * d_predicted_output * lr;
        output_bias += d_predicted_output.sum() * lr;
        hidden_weights += inputs.transpose() * d_hidden_layer * lr;
        hidden_bias += d_hidden_layer.sum() * lr;
    }

    println!("Final hidden weights: {:?}", hidden_weights);
    println!("Final hidden bias: {:?}", hidden_bias);
    println!("Final output weights: {:?}", output_weights);
    println!("Final output bias: {:?}", output_bias);
}
