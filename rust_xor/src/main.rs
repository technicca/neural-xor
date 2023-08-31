// Create a v2 of this net (tanh, no bias neuron but bias values for each neuron, weights initialized between 0.1 and 0.5, biases initialized with 0.5 each, 1.000 training data sets from 0.001 to 2.0 and activation normalization (input/activation of all but input layer neurons are divided by the amount of neurons in the parent layer), 1-5-1 neurons) I tried your problem and got a 95% averaged accuracy in less than 2.000 epochs every time with a learning rate of 0.1.
// That one should be able to solve the xor in 100 epochs with lr of 0.2 and 2,5,1 neuron structure

use ndarray::{Array2, arr2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// Correct Sigmoid function - using expit because it's more stable for large inputs
fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    let expit = |x: f64| 1.0 / (1.0 + (-x).exp());
    z.mapv(expit)
}

fn sigmoid_prime(z: &Array2<f64>) -> Array2<f64> {
    let sig = sigmoid(z);
    &sig * &(1.0 - &sig)
}

fn main() {
    let input_neurons = 2;
    let hidden_neurons = 5;
    let output_neurons = 1;
    let epochs = 10000;
    let learning_rate: f64 = 0.8;

    // XOR dataset
    let inputs = arr2(&[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]);

    let outputs = arr2(&[[0.0], [1.0], [1.0], [0.0]]);

    // Initialize weights and biases
    let mut rng = rand::thread_rng();
    let mut weights1 = Array2::random_using((input_neurons, hidden_neurons), Uniform::new(-1.0, 1.0), &mut rng); 
    let mut weights2 = Array2::random_using((hidden_neurons, output_neurons), Uniform::new(-1.0, 1.0), &mut rng);
    let mut bias1 = Array2::random_using((1, hidden_neurons), Uniform::new(-1.0, 1.0), &mut rng);
    let mut bias2 = Array2::random_using((1, output_neurons), Uniform::new(-1.0, 1.0), &mut rng);

    // Training loop
    for _ in 0..epochs {
        // Forward propagation
        let layer1 = sigmoid(&(inputs.dot(&weights1) + &bias1));
        let layer2 = sigmoid(&(layer1.dot(&weights2) + &bias2));

        // Backpropagation
        let layer2_error = outputs.clone() - &layer2;
        let layer2_delta = layer2_error * sigmoid_prime(&layer2);

        let layer1_error = layer2_delta.dot(&weights2.t());
        let layer1_delta = layer1_error * sigmoid_prime(&layer1);

        // Update weights and biases
        weights2 = weights2 + layer1.t().dot(&layer2_delta) * learning_rate;
        weights1 = weights1 + inputs.t().dot(&layer1_delta) * learning_rate;
        bias2 = bias2 + layer2_delta.sum_axis(ndarray::Axis(0)) * learning_rate;
        bias1 = bias1 + layer1_delta.sum_axis(ndarray::Axis(0)) * learning_rate;
    }

    // Show output after training
    let layer1 = sigmoid(&(inputs.dot(&weights1) + &bias1));
    let layer2 = sigmoid(&(layer1.dot(&weights2) + &bias2));
    println!("{:?}", layer2);
}
