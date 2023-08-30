#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define the structure of a neuron
typedef struct {
    double weights[2];
    double bias;
} Neuron;

// Define the structure of a network
typedef struct {
    Neuron hidden[2];
    Neuron output;
} Network;

// Define the sigmoid function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Define the derivative of the sigmoid function
double sigmoid_prime(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

// Define the feedforward function
double feedforward(Network* network, double inputs[2]) {
    double hidden_outputs[2];
    for (int i = 0; i < 2; i++) {
        hidden_outputs[i] = sigmoid(network->hidden[i].weights[0] * inputs[0] + network->hidden[i].weights[1] * inputs[1] + network->hidden[i].bias);
    }
    return sigmoid(network->output.weights[0] * hidden_outputs[0] + network->output.weights[1] * hidden_outputs[1] + network->output.bias);
}

// Define the train function
void train(Network* network, double inputs[2], double target) {
    // Feedforward
    double hidden_outputs[2];
    for (int i = 0; i < 2; i++) {
        hidden_outputs[i] = sigmoid(network->hidden[i].weights[0] * inputs[0] + network->hidden[i].weights[1] * inputs[1] + network->hidden[i].bias);
    }
    double output = sigmoid(network->output.weights[0] * hidden_outputs[0] + network->output.weights[1] * hidden_outputs[1] + network->output.bias);

    // Calculate output error
    double output_error = target - output;

    // Calculate output gradient
    double output_gradient = output_error * sigmoid_prime(output);

    // Adjust output weights and bias
    for (int i = 0; i < 2; i++) {
        network->output.weights[i] += output_gradient * hidden_outputs[i];
    }
    network->output.bias += output_gradient;

    // Calculate hidden errors
    double hidden_errors[2];
    for (int i = 0; i < 2; i++) {
        hidden_errors[i] = network->output.weights[i] * output_error;
    }

    // Calculate hidden gradients and adjust hidden weights and biases
    for (int i = 0; i < 2; i++) {
        double hidden_gradient = hidden_errors[i] * sigmoid_prime(hidden_outputs[i]);
        for (int j = 0; j < 2; j++) {
            network->hidden[i].weights[j] += hidden_gradient * inputs[j];
        }
        network->hidden[i].bias += hidden_gradient;
    }
}

int main() {
    // Initialize network
    Network network = {0};

    // Define training data
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4] = {0, 1, 1, 0};

    // Train network
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 4; j++) {
            train(&network, inputs[j], targets[j]);
        }
    }

    // Test network
    for (int i = 0; i < 4; i++) {
        printf("XOR(%d, %d) = %f\n", (int)inputs[i][0], (int)inputs[i][1], feedforward(&network, inputs[i]));
    }

    return 0;
}
