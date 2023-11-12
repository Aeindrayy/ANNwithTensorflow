import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder


digits = load_digits()
images = digits.data
labels = digits.target
print(images.shape)

# Display a sample image
plt.gray()
plt.matshow(images[90].reshape(8, 8))
plt.show()

# Reshape images
images = images.reshape(images.shape[0], -1)
print(images.shape)

#change float32 type and {0,1} scaling
images = images.astype('float32') / 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False, categories='auto')
labels = labels.reshape(-1, 1)
onehot_labels = encoder.fit_transform(labels)

for i, digit in enumerate(digits.target_names):
    print(f"{digit}: {onehot_labels[i]}")

# Minibatch generator 
def generate_minibatch(images, labels, minibatch_size):
    dataset = list(zip(images, labels))
    while True:
        np.random.shuffle(dataset)

        for i in range(0, len(dataset), minibatch_size):
            minibatch = dataset[i:i + minibatch_size]
            minibatch_images, minibatch_labels = zip(*minibatch)

            minibatch_images = np.array(minibatch_images)
            minibatch_labels = np.array(minibatch_labels)

            yield minibatch_images, minibatch_labels

# Print minibatch shapes
minibatch_size = 40
minibatch_generator = generate_minibatch(images, labels, minibatch_size)

for i in range(minibatch_size):
    minibatch_images, minibatch_labels = next(minibatch_generator)
    print(f"Minibatch Images Shape: {minibatch_images.shape}, Minibatch Labels Shape: {minibatch_labels.shape}")

#sigmoid activation 
class SigmoidActivation:
    def __call__(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backward(self, activated_outputs):
        return activated_outputs * (1 - activated_outputs)

#softmax activation 
class SoftmaxActivation:
    def __call__(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        softmax_outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.activated_outputs = softmax_outputs
        return softmax_outputs

    def backward(self, error_signal):
        softmax_derivative = self.activated_outputs * (1 - self.activated_outputs)
        grad_weighted_sum = error_signal * softmax_derivative
        return grad_weighted_sum

# MLP class
class SimpleMLPLayer:
    def __init__(self, activation_function, num_units, input_size):
        self.activation_function = activation_function
        self.num_units = num_units
        self.input_size = input_size
        self.weights = np.random.normal(loc=0.0, scale=0.2, size=(input_size, num_units))
        self.bias = np.zeros((1, num_units))

        self.inputs = None
        self.weighted_sum = None
        self.activated_outputs = None

        self.grad_weights = None
        self.grad_bias = None

    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.activated_outputs = self.activation_function(self.weighted_sum)
        return self.activated_outputs

    def backward(self, error_signal):
        sigmoid_derivative = self.activation_function.backward(self.activated_outputs)
        grad_weighted_sum = error_signal * sigmoid_derivative

        self.grad_weights = np.dot(self.inputs.T, grad_weighted_sum)
        self.grad_bias = np.sum(grad_weighted_sum, axis=0, keepdims=True)

        grad_inputs = np.dot(grad_weighted_sum, self.weights.T)
        return grad_inputs

# Full MLP class
class SimpleMLP:
    def __init__(self, layer_sizes, activation_functions):
        assert len(layer_sizes) == len(activation_functions) + 1, "Mismatch in the number of layers and activation functions."
        self.layers = [SimpleMLPLayer(activation_function=af, num_units=num_units, input_size=input_size)
                       for af, num_units, input_size in zip(activation_functions, layer_sizes[1:], layer_sizes[:-1])]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

# Cross-Entropy Loss 
class CrossEntropyLoss:
    def __call__(self, predicted_probs, true_labels):
        predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)
        return -np.sum(true_labels * np.log(predicted_probs)) / len(true_labels)

    def backward(self, predicted_probs, true_labels):
        return predicted_probs - true_labels

# Training function ( this part is not working)
def train_simple_mlp(mlp, train_images, train_labels, epochs, minibatch_size, learning_rate):
    loss_history = []

    minibatch_generator = generate_minibatch(train_images, train_labels, minibatch_size)
    loss_function = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for _ in range(len(train_images) // minibatch_size):
            minibatch_images, minibatch_labels = next(minibatch_generator)

            predictions = mlp.forward(minibatch_images)
            loss = loss_function(predictions, minibatch_labels)
            total_loss += loss

            loss_grad = loss_function.backward(predictions, minibatch_labels)
            mlp.backward(loss_grad)

            for layer in mlp.layers:
                layer.weights -= learning_rate * layer.grad_weights
                layer.bias -= learning
        avg_loss = total_loss / (len(train_data) // minibatch_size)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

        if validation_data is not None and validation_labels is not None:
            validation_predictions = mlp.forward(validation_data)
            validation_loss = loss_function(validation_predictions, validation_labels)
            print(f"Validation Loss: {validation_loss}")

    plt.plot(range(1, epochs + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()
