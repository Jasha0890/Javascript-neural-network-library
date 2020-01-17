/*
* Library to build a neural network to
* feedforward and backpropagate the data
*
* 
* Sigmoidal activation function
*/
function sigmoid(x) {
  
  return 1 / (1 + Math.exp(-x));
}

function delta_sigmoid(y) {
  
  // Updates the sigmoid function's value after calculating the error
  return y * (1 - y);
}

/*
* Create a class called neural network with
* a constructor including the paramaters needed to build
* the multi layer perceptron network
*/
class NeuralNetwork {
  
  constructor(inputNodes, hiddenNodes, outputNodes) {
    
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    this.weights_ItoH = new Matrix(this.hiddenNodes, this.inputNodes);
    this.weights_HtoOut = new Matrix(this.outputNodes, this.hiddenNodes);
    this.weights_ItoH.randomize();
    this.weights_HtoOut.randomize();

    this.bias_hidden = new Matrix(this.hiddenNodes, 1);
    this.bias_output = new Matrix(this.outputNodes, 1);
    this.bias_hidden.randomize();
    this.bias_output.randomize();
    // Set the learning rate to a random number between 0 (included) and 1 (excluded)
    this.learning_rate = Math.random();
  }

  feed_forward(input_array) {

    // Generating the Hidden Outputs (hidden to output)
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ItoH, inputs);
    hidden.add(this.bias_hidden);
    // Sigmoidal activation function
    hidden.map(sigmoid);
    // Generating the output's output (y)
    let output = Matrix.multiply(this.weights_HtoOut, hidden);
    output.add(this.bias_output);
    output.map(sigmoid);

    
    return output.toArray();
  }

/*
* training the current array to generate new values
* and produce the right input to add to a new array (backpropagation)
*/
  train(input_array, target_array) {
    
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ItoH, inputs);
    hidden.add(this.bias_hidden);
    // Sigmoidal activation function
    hidden.map(sigmoid);
    // Generating the output's output (y)
    let output = Matrix.multiply(this.weights_HtoOut, hidden);
    output.add(this.bias_output);
    output.map(sigmoid);
    // Convert array to a matrix of desired outputs
    let targets = Matrix.fromArray(target_array);
    // Calculate the error
    // Error formula = targetOutput - currentOutput
    let output_errors = Matrix.subtract(targets, output);
    // Gradient descent formula = outputs * (1 - outputs);
    // Calculate gradient descent
    let gradients = Matrix.map(output, delta_sigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);


    // Calculate deltas hidden to output (the changes to the weights to get the right output)
    let hidden_T = Matrix.transpose(hidden);
    let weight_HtoOut_deltas = Matrix.multiply(gradients, hidden_T);
    // Modifiying the weights adding the deltas values
    this.weights_HtoOut.add(weight_HtoOut_deltas);
    // Modifiying the bias adding the deltas value (gradient)
    this.bias_output.add(gradients);
    // Calculate the hidden layer errors
    let weights_HtoOut_t = Matrix.transpose(this.weights_HtoOut);
    let hidden_errors = Matrix.multiply(weights_HtoOut_t, output_errors);
    // Calculate hidden gradient
    let hidden_gradient = Matrix.map(hidden, delta_sigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);
    // Calculate deltas input to hidden (the changes to the weights to get the right output)
    let inputs_T = Matrix.transpose(inputs);
    let weights_ItoH_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ItoH.add(weights_ItoH_deltas);
    // Modifiying the bias adding the deltas value (hidden gradient)
    this.bias_hidden.add(hidden_gradient);

    
  }

}