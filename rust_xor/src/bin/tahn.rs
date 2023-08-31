use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn tanh(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|x| x.tanh())
}

fn tanh_prime(z: &Array2<f64>) -> Array2<f64> {
    1.0 - z.mapv(|x| x.tanh().powi(2))
}

fn main() {
    let input_neurons = 1;
    let hidden_neurons = 5;
    let output_neurons = 1;
    let epochs = 100;
    let learning_rate: f64 = 0.2;

    let inputs = Array1::linspace(0.001, 2.0, 1000).into_shape((1000, 1)).unwrap();

    let outputs = inputs.mapv(|x| if x <= 1.0 { 0.0 } else { 1.0 });

    // Initialize weights and biases
    let mut rng = rand::thread_rng();
    let mut weights1 = Array2::random_using((input_neurons, hidden_neurons), Uniform::new(0.1, 0.5), &mut rng); 
    let mut weights2 = Array2::random_using((hidden_neurons, output_neurons), Uniform::new(0.1, 0.5), &mut rng);
    let mut bias1 = Array2::from_elem((1, hidden_neurons), 0.5);
    let mut bias2 = Array2::from_elem((1, output_neurons), 0.5);

    for _ in 0..epochs { // manin loop
        // Forward propagation
        let layer1 = tanh(&(inputs.dot(&weights1) / input_neurons as f64 + &bias1));
        let layer2 = tanh(&(layer1.dot(&weights2) / hidden_neurons as f64 + &bias2));

        // Backpropagation
        let layer2_error = outputs.clone() - &layer2;
        let layer2_delta = layer2_error * tanh_prime(&layer2);

        let layer1_error = layer2_delta.dot(&weights2.t());
        let layer1_delta = layer1_error * tanh_prime(&layer1);

        // Update weights and biases
        weights2 = weights2 + layer1.t().dot(&layer2_delta) * learning_rate;
        weights1 = weights1 + inputs.t().dot(&layer1_delta) * learning_rate;
        bias2 = bias2 + layer2_delta.sum_axis(ndarray::Axis(0)) * learning_rate;
        bias1 = bias1 + layer1_delta.sum_axis(ndarray::Axis(0)) * learning_rate;
    }

    let layer1 = tanh(&(inputs.dot(&weights1) / input_neurons as f64 + &bias1));
    let layer2 = tanh(&(layer1.dot(&weights2) / hidden_neurons as f64 + &bias2));
    println!("{:?}", layer2);
}
