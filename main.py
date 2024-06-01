import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Layer:

    def __init__(self, input_size: int, output_size: int, is_input_layer: bool):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = tf.Variable(np.random.rand(input_size, output_size).astype(np.float32), dtype=tf.float32)
        self.is_input_layer = is_input_layer
        #if is_input_layer:
        #    self.weights = tf.reshape(self.weights, [self.weights.shape[1], self.weights.shape[0]])


    def feed_forward(self, inputs):
        x = tf.cast(inputs, tf.float32)
        # inputs (4, 2)
        #print("Input: ", x)
        #print("Weights", self.weights)
        summed = tf.matmul(x, self.weights)
        self.output = self.activation_function(summed)
        # Should be Data x Neurons (4, 2)
        return self.output
    
    def activation_function(self, x):
        const_a = tf.constant(1.0)
        const_b = tf.constant(2.0)
        pow = -tf.math.exp(const_a) * x + (tf.math.exp(const_a)/const_b)
        
        return const_a / (const_a + tf.math.exp(pow)) 

class SigmoidLayer(Layer):

    def activation_function(self, x):
        const = tf.constant(1.0)
        return const / (const + tf.math.exp(-x))

class ReLULayer(Layer):

    def activation_function(self, x):
        return tf.where(x > 0, x, 0)

class Layer2(Layer):

    def activation_function(self, x):
        const_a = tf.constant(1.0)
        const_b = tf.constant(2.0)
        const_c = tf.constant(10.0)
        const_d = tf.constant(4.0)
        pow = -tf.math.exp(const_a) * x * const_c + (const_d)
        
        return const_a / (const_a + tf.math.exp(pow))

data = np.array([[1,1],[1,0],[0,1],[0,0]])
input_layer = Layer(2, 4, True)
hidden_layer = Layer(4, 2, False)
output_layer = Layer(2, 2, False)
layers = [input_layer, hidden_layer, output_layer]

def back_prop(inp, layers):
        
    x_a = []
    y_a = []

    EPOCHS = 1500 
    LEARNING_RATE = 1 
    for epoch in range(EPOCHS):
        output = inp
        with tf.GradientTape(persistent=True) as Tape:
            for layer in layers:
                Tape.watch(layer.weights)
            weights = []
            for layer in layers:
                output = layer.feed_forward(output)
                weights.append(layer.weights)
            desired = tf.Variable([[1, 0],[0, 1],[0, 1],[1, 0]], dtype=tf.float32)
            #loss = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits([[1, 0],[0, 1],[0, 1],[1, 0]], output))
            #print("Output: ", output)
            
            loss = -tf.reduce_mean(tf.square(tf.subtract(output,desired)))
            print("Loss: ", loss)
            gradients = Tape.gradient(loss, weights)

            y_a.append(loss)
            x_a.append(epoch)

            for index, layer_weight_array in enumerate(layers):
                layers[index].weights = tf.Variable(tf.add(layers[index].weights, gradients[index]*LEARNING_RATE), dtype=tf.float32)
                #print("New Weights: ", layers[index].weights)

    plt.xlabel("Epoch")
    plt.ylabel("-Loss (MSE)")
    plt.plot(x_a, y_a)
    
    return output


plt.title("New Loss Func")
output = back_prop(data, layers)
print(output)
plt.show()

