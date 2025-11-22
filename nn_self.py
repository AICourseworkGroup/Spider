import numpy as np
from random import random

# because a lot of data will be needed, we will use a class approach.
class Full_NN(object):
    """
    A simple feed forward neural network with back propagation and gradient descent.

    args:
        X: Number of input features
        HL: List of hidden layer sizes
        Y: Number of output features 

    """


    # A Multi Layer Neural Network class. We use this as for the way we need to handle the
    # variables is better suited.
    def __init__(self, X=2, HL=[2,2], Y=2):  # a constructor for some default values.
        self.X, self.HL, self.Y = X, HL, Y  # inputs, hidden layers, outputs

        """
        Initialize the neural network, where we set up the weights, derivatives, and outputs for each layer.



        args:
            L = [X] + HL + [Y]: Total number of layers including input, hidden, and output layers (Creates repensentation of the network in specified format)
            W = []: Initialize a weight array
            

        """

        # Setting up some class variables for our inputs.
        L = [X] + HL + [Y] 
        W = []

        # we want to be able go to the next layer up so we set one minus
        for i in range(len(L)-1):  
            # Use Xavier initialization: weights scaled by sqrt(1 / input_size)
            # This prevents vanishing/exploding gradients in deep networks
            w = np.random.randn(L[i], L[i+1]) * np.sqrt(2.0 / L[i])
            W.append(w)  # add the new values to the array.
            
        self.W = W  # assign the list of weights to the class variable.

        Der = []  # initialize a derivative array. This are needed to calculate the back propagation. they are the derivatives of the activation function.
        for i in range(len(L)-1):  # same reason as above for every line
            d = np.zeros((L[i], L[i+1]))  # we don't need random values, just to have them ready to be used. we fill up with zeros.
            Der.append(d)
        self.Der = Der
        # we will be passing these here as that way the class variable will keep them for us until we need them.
        
        out = []  # initialize output array
        for i in range(len(L)):  # We don't need to go +1. The outputs are straight forward.
            o = np.zeros(L[i])  # we don't need random values, just to have them ready to be used. we fill up with zeros.
            out.append(o)
        self.out = out

    def FF(self, x):  # This method will run the network forward
        out = np.array(x, dtype=np.float32)  # convert the input to a numpy array
        self.out[0] = out  # begin the linking of outputs to the class variable for back propagation. (begin with the input layer.
        for i, w in enumerate(self.W):  # go through (iterate) the network layers via the weights variable
            Xnext = np.dot(out, w)  # calculate product between weights and output for the next output
            # Apply sigmoid activation to all layers EXCEPT the last one
            if i < len(self.W) - 1:
                out = self.sigmoid(Xnext)  # use the activation function for hidden layers
            else:
                out = Xnext  # No activation for output layer (linear output for regression)
            self.out[i+1] = out  # pass the result to the class variable to preserve for later (when we do the back propagation.
        return out  # return the outputs of the layers.

    def BP(self, Er):
        """
        Back Propagation method, using the Output Error (Er) to iterate backwards through the layers and calculate the errors needed to update the Weights and returns the final error of the input.

        This is based on the equations for back propagation.
            dE/dW_i = (y - y_hat) * S'(z_i) * x_i
            S'(z_i) = S(z_i) * (1 - S(z_i)) for sigmoid
            S'(z_i) = 1 for linear (output layer)
            z_{i+1} = a_i * W_i

        args:
            D = Er : Linear activation derivative is 1, so just use error directly 
            Er: The error at the output layer (target - output)
            out: returns the output for the previous layer (in reverse order)

            D_fixed = D.reshape(D.shape[0], -1).T: Turns Delta into an array of appropriate size
            this_out = self.out[i]: current layer output.
            this_out = this_out.reshape(this_out.shape[0], -1): reshape as before to get column array suitable for the multiplication we need.
            self.Der[i] = np.dot(this_out, D_fixed): Calculate the derivative for weight update and store in class variable.

            Er = np.dot(D, self.W[i].T): This essentially back propagates the next error we need for the next iteration. This error term
            is part of the dE/DWi equation for the next layer down in the back propagation, and we pass it on after calculating it in this iteration.

        """
        
        for i in reversed(range(len(self.Der))):  # this is a trick allowed by Python, we can go back in reverse and essentially go backwards into the network.
            
            out = self.out[i+1] 
            
            # For output layer (last layer), derivative is 1 (linear activation)
            # For hidden layers, use sigmoid derivative

            if i == len(self.Der) - 1:
                # Linear activation derivative is 1, so just use error directly
                D = Er  
            else:
                # Apply sigmoid derivative for hidden layers
                D = Er * self.sigmoid_Der(out)  
            
            D_fixed = D.reshape(D.shape[0], -1).T
            this_out = self.out[i]
            this_out = this_out.reshape(this_out.shape[0], -1)
            self.Der[i] = np.dot(this_out, D_fixed)
            Er = np.dot(D, self.W[i].T) 

    def train_nn(self, x, target, epochs, lr):  # training the network. The x is an array, the target is an array the epochs is a number and the lr is a number.

        """
        Train the neural network using back propagation and gradient descent. The training loop occurs over the specified number of epochs.

        args:
            epochs: Number of training epochs
            S_errors: Variable to carry the error we need to report to the user
            x: List of input poses (random poses)
            target: List of target poses (GA poses)
            lr: Learning rate for weight updates
            output = self.FF(input): Forward pass to get the network output for the given input
            e = t - output: Calculate the error between target and output
            MSQE: Mean Squared Quadratic Error
        """

        for i in range(epochs):
            S_errors = 0  
            for j, input in enumerate(x):

                # Convert input and target to numpy arrays
                t = np.array(target[j], dtype=np.float32)

                output = self.FF(input) 
                e = t - output

                # Do gradient descent and back propagation
                self.BP(e)
                self.GD(lr)

                # update the overall error to show the user
                S_errors += (t - output) ** 2

            # Print mean squared quadratic error (MSQE) for this epoch so user can track training
            epoch_loss = np.average(S_errors)
            print(f"Epoch {i+1}/{epochs} MSQE: {epoch_loss}")


    def GD(self, lr=0.05): 
        """
        Update weights and learning rate, with gradient descent.

        args:
            lr: Learning rate for weight updates
            W: List of weight matrices between layers
            Der: List of derivatives for each weight matrix

        """

        for i in range(len(self.W)):  # Iterates through the weights
            W = self.W[i]
            Der = self.Der[i]
            W += Der * lr 

    def sigmoid(self, x): 
        """
        Sigmoid activation function is used to introduce non-linearity into the network.

        args:
            x: Input value to the sigmoid function
        """

        y = 1.0 / (1 + np.exp(-x))
        return y

    def sigmoid_Der(self, x):
        """
        Derivative of the sigmoid function

        args:
            x: Input value to calculate the derivative
        
        """

        sig_der = x * (1.0 - x)
        return sig_der

    def msqe(self, t, output):
        """
        Calculate Mean Squared Quadratic Error between target and output.

        args:
            t: Target output
            output: Neural network output
        """

        msq = np.average((t - output) ** 2)
        return msq


def randAngGen():
    """ 
    Generate a random angle between -45 and 45 degrees in radians for spider joint angles. 

    args:
        None
    """

    angle = (random() * np.pi / 2) - (np.pi / 4) 
    return angle

def genRanPoses(popSize=3000):
    """ 
    Generate random poses for neural network training, by returning a list of lists that contains random poses from each joint.
    where each leg has 3 joints (a, b, c) and there are 8 legs (L1, L2, L3, L4, R4, R3, R2, R1)
     
    args:
        popSize: Number of random poses to generate
       """
    
    poses = []
    for _ in range(popSize):
        l1 = [randAngGen(), randAngGen(), randAngGen()]
        l2 = [-randAngGen(), randAngGen(), randAngGen()]
        l3 = [randAngGen(), randAngGen(), randAngGen()]
        l4 = [-randAngGen(), randAngGen(), randAngGen()]
        r4 = [randAngGen(), randAngGen(), randAngGen()]
        r3 = [-randAngGen(), randAngGen(), randAngGen()]
        r2 = [randAngGen(), randAngGen(), randAngGen()]
        r1 = [-randAngGen(), randAngGen(), randAngGen()]
        angles = l1 + l2 + l3 + l4 + r4 + r3 + r2 + r1
        poses.append(angles)
    return poses

