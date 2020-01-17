# Multi-layer perceptron neural netowrk JS library

Neural network javascript library to help build a multi layer perceptron neural network. This library is inspired by [Toy-Neural-Network](https://github.com/CodingTrain/Toy-Neural-Network-JS).

A different version built with Matlab : [Matlab MLP neural network](https://github.com/Jasha0890/Matlab-neural-network/blob/master/MLP.m).

## Features

- Neural Network with variable amounts of inputs, hidden nodes and outputs
- Multiple hidden layers
- Activation functions: Sigmoid
- Adjustable learning rate
- Fully connected


### Use the library

* **Initialize** Neural Network

```javascript
// Build the neural network with 3 layers (input, hidden, output) and as many nodes as needed per layer
constructor(inputNodes, hiddenNodes, outputNodes) {
    
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
// initializes the neural network with random weights
    this.weights_ItoH.randomize();
    this.weights_HtoOut.randomize();
```

* **Activation function** 

```javascript
//Sigmoidal activation function
function sigmoid(x) {
  
  return 1 / (1 + Math.exp(-x));
}

```

* Performing **feed forward**


```javascript
//Passing the inputs into the layers of the neural network structure to produce a final output
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

```

* **Training** Calculate the error

```javascript

// Error formula = targetOutput - currentOutput
let output_errors = Matrix.subtract(targets, output);
// Gradient descent formula = outputs * (1 - outputs);
// Calculate gradient descent
let gradients = Matrix.map(output, delta_sigmoid);
gradients.multiply(output_errors);
gradients.multiply(this.learning_rate);

// Calculate the hidden layer errors
let weights_HtoOut_t = Matrix.transpose(this.weights_HtoOut);
let hidden_errors = Matrix.multiply(weights_HtoOut_t, output_errors);

```

* **Training** Calculate the deltas

```javascript
// Calculate deltas hidden to output (the changes to the weights to get the right output)
let hidden_T = Matrix.transpose(hidden);
let weight_HtoOut_deltas = Matrix.multiply(gradients, hidden_T);

// Calculate deltas input to hidden (the changes to the weights to get the right output)
let inputs_T = Matrix.transpose(inputs);
let weights_ItoH_deltas = Matrix.multiply(hidden_gradient, inputs_T);

```

* **Training** Modify weights and bias using deltas

```javascript
// Modifiying the weights adding the deltas values
this.weights_HtoOut.add(weight_HtoOut_deltas);
// Modifiying the bias adding the deltas value (gradient)
this.bias_output.add(gradients);

// Calculate hidden gradient
let hidden_gradient = Matrix.map(hidden, delta_sigmoid);
hidden_gradient.multiply(hidden_errors);
hidden_gradient.multiply(this.learning_rate);

// Modifiying the bias adding the deltas value (hidden gradient)
this.bias_hidden.add(hidden_gradient);

```


