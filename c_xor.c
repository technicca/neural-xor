#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 8
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.1
#define EPOCHS 20000

// Define the structure of a neuron
typedef struct {
    double weights[INPUT_NEURONS];
    double bias;
} Neuron;

// Define the structure of a network
typedef struct {
    Neuron hidden[HIDDEN_NEURONS];
    Neuron output[HIDDEN_NEURONS];
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
double feedforward(Network* network, double inputs[INPUT_NEURONS]) {
    double hidden_outputs[HIDDEN_NEURONS];
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        hidden_outputs[i] = sigmoid(network->hidden[i].weights[0] * inputs[0] + network->hidden[i].weights[1] * inputs[1] + network->hidden[i].bias);
    }
    return sigmoid(network->output[0].weights[0] * hidden_outputs[0] + network->output[0].weights[1] * hidden_outputs[1] + network->output[0].bias);
}

// Define the train function
void train(Network* network, double inputs[INPUT_NEURONS], double target) {
    // Feedforward
    double hidden_outputs[HIDDEN_NEURONS];
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        hidden_outputs[i] = sigmoid(network->hidden[i].weights[0] * inputs[0] + network->hidden[i].weights[1] * inputs[1] + network->hidden[i].bias);
    }
    double output = sigmoid(network->output[0].weights[0] * hidden_outputs[0] + network->output[0].weights[1] * hidden_outputs[1] + network->output[0].bias);

    // Calculate output error
    double output_error = target - output;

    // Calculate output gradient
    double output_gradient = output_error * sigmoid_prime(output);

    // Adjust output weights and bias
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        network->output[0].weights[i] += output_gradient * hidden_outputs[i] * LEARNING_RATE;
    }
    network->output[0].bias += output_gradient * LEARNING_RATE;

    // Calculate hidden errors
    double hidden_errors[HIDDEN_NEURONS];
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        hidden_errors[i] = network->output[0].weights[i] * output_error;
    }

    // Calculate hidden gradients and adjust hidden weights and biases
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        double hidden_gradient = hidden_errors[i] * sigmoid_prime(hidden_outputs[i]);
        for (int j = 0; j < INPUT_NEURONS; j++) {
            network->hidden[i].weights[j] += hidden_gradient * inputs[j] * LEARNING_RATE;
        }
        network->hidden[i].bias += hidden_gradient * LEARNING_RATE;
    }
}

int main() {
    // Initialize network
    Network network = {0};

    // Initialize random number generator
    srand(time(NULL));

    // Initialize weights and biases to random value
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        for (int j = 0; j < INPUT_NEURONS; j++) {
            network.hidden[i].weights[j] = (double)rand() / RAND_MAX;
        }
        network.hidden[i].bias = (double)rand() / RAND_MAX;
        network.output[0].weights[i] = (double)rand() / RAND_MAX;
    }
    network.output[0].bias = (double)rand() / RAND_MAX;

    // Define training data
    double inputs[4][INPUT_NEURONS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4] = {0, 1, 1, 0};

    // Train network
    for (int i = 0; i < EPOCHS; i++) {
        for (int j = 0; j < 4; j++) {
            train(&network, inputs[j], targets[j]);
        }
    }

    // Print final weights and biases
    printf("Final hidden weights:\n");
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        printf("Neuron %d: %f, %f\n", i, network.hidden[i].weights[0], network.hidden[i].weights[1]);
    }
    printf("Final hidden biases:\n");
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        printf("Neuron %d: %f\n", i, network.hidden[i].bias);
    }
    printf("Final output weights: %f, %f\n", network.output[0].weights[0], network.output[0].weights[1]);
    printf("Final output bias: %f\n", network.output[0].bias);

    // Test network
    printf("\nOutput from neural network after %d epochs:\n", EPOCHS);
    for (int i = 0; i < 4; i++) {
        printf("XOR(%d, %d) = %f\n", (int)inputs[i][0], (int)inputs[i][1], feedforward(&network, inputs[i]));
    }

    return 0;
}
