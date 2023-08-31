use ndarray::{Array1, Array2, arr1, arr2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + (-z).mapv(f64::exp))
}
fn sigmoid_prime(z: &Array2<f64>) -> Array2<f64> {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn main() {
    let inputs = arr2(&[
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ]);

    let outputs = arr1(&[0., 1., 1., 0.]);

    // weights and biases
    let mut weights1 = Array2::random((2, 2), Uniform::new(-1.0, 1.0));
    let mut weights2 = Array2::random((2, 1), Uniform::new(-1.0, 1.0));
    let mut bias1 = Array2::random((1, 2), Uniform::new(-1.0, 1.0));
    let mut bias2 = Array2::random((1, 1), Uniform::new(-1.0, 1.0));

    
    for _ in 0..50000 { // Main training loop
        // Forward propagation
        let layer1 = sigmoid(&(inputs.dot(&weights1) + &bias1));
        let layer2 = sigmoid(&(layer1.dot(&weights2) + &bias2));

        // Backpropagation
        let layer2_error = outputs.clone().into_shape((4, 1)).unwrap() - &layer2;
        let layer2_delta = layer2_error * sigmoid_prime(&layer2);

        let layer1_error = layer2_delta.dot(&weights2.t());
        let layer1_delta = layer1_error * sigmoid_prime(&layer1);

        // Update weights and biases
        weights2 = weights2 + layer1.t().dot(&layer2_delta);
        weights1 = weights1 + inputs.t().dot(&layer1_delta);
        bias2 = bias2 + layer2_delta.sum_axis(ndarray::Axis(0));
        bias1 = bias1 + layer1_delta.sum_axis(ndarray::Axis(0));
    }

    // Out
    let layer1 = sigmoid(&(inputs.dot(&weights1) + &bias1));
    let layer2 = sigmoid(&(layer1.dot(&weights2) + &bias2));
    println!("{:?}", layer2);
}