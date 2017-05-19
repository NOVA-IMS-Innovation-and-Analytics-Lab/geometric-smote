
"""
This module contains the classes for Generative 
Adversarial Networks (GAN) and Conditional Generative 
Adversarial Networks (CGAN).
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab as pl
import random
from matplotlib import colors as mcolors
from math import sqrt
from IPython import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


OPTIMIZER = tf.train.AdamOptimizer()

def bind_columns(tensor1, tensor2):
    """Column binds the tensors if the second tensor exists, 
    else returns the first tensor."""
    if tensor2 is None:
        return tensor1
    return tf.concat(axis=1, values=[tensor1, tensor2])

def initialize_model_parameters(n_input_units, n_output_units, initialization_choice):
    shape = [n_input_units, n_output_units] if n_output_units is not None else [n_input_units]
    initialization_types = {'xavier': tf.random_normal(shape=shape, stddev=1. / tf.sqrt(n_input_units / 2.)), 
                            'normal': tf.random_normal(shape=shape), 
                            'zeros': tf.zeros(shape=shape)}
    return initialization_types[initialization_choice]

def initialize_model(model_layers, input_layer_correction, weights_initialization_choice, bias_initialization_choice):
    """Initializes variables for the model parameters and 
    a placeholder for the input data."""
    model_parameters = {}
    for layer_index in range(len(model_layers) - 1):
        model_parameters['W' + str(layer_index)] = tf.Variable(initialize_model_parameters(model_layers[layer_index][0], model_layers[layer_index + 1][0], weights_initialization_choice))
        model_parameters['b' + str(layer_index)] = tf.Variable(initialize_model_parameters(model_layers[layer_index + 1][0], None, bias_initialization_choice))
    input_data_placeholder = tf.placeholder(tf.float32, shape=[None, model_layers[0][0] - input_layer_correction])
    return input_data_placeholder, model_parameters
    
def output_logits_tensor(input_tensor, model_layers, model_parameters):
    """Returns the output logits of a model given its parameters and 
    an input tensor."""
    output_tensor = input_tensor
    for layer_index in range(len(model_layers) - 1):
        logit_tensor = tf.matmul(output_tensor, model_parameters['W' + str(layer_index)]) + model_parameters['b' + str(layer_index)]
        activation_function = model_layers[layer_index + 1][1]
        if activation_function is not None:
            output_tensor = activation_function(logit_tensor)
        else:
            output_tensor = logit_tensor
    return output_tensor

def sample_Z(n_samples, n_features):
    """Samples the elements of a (n_samples, n_features) shape 
    matrix from a uniform distribution in the [-1, 1] interval."""
    return np.random.uniform(-1., 1., size=[n_samples, n_features]).astype(np.float32)

def sample_y(n_samples, n_y_features, class_label):
    """Returns a matrix of (n_samples, n_y_features) shape using 
    one-hot encoding for the class label. """
    if n_y_features > 2:
        y = np.zeros(shape=[n_samples, n_y_features], dtype='float32')
        y[:, class_label] = 1.
    else:
        y = np.zeros([n_samples, 1], dtype='float32') if class_label == 0 else np.ones([n_samples, 1], dtype='float32')
    return y
    
def cross_entropy(logits, positive_class_labels):
    """Returns the loss function of the discriminator or generator 
    for  positive or negative class labels."""
    if positive_class_labels:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))
    return loss

def accuracy(logits, positive_class_labels):
    if positive_class_labels:
        result = tf.reduce_mean(tf.cast(tf.greater_equal(tf.sigmoid(logits), 0.5), tf.float32))
    else:
        result = tf.reduce_mean(tf.cast(tf.less(tf.sigmoid(logits), 0.5), tf.float32))
    return result

def define_optimization(optimizer, loss, model_parameters):
    """Defines the optimization problem for a given optimizer, 
    loss function and model parameters."""
    return optimizer.minimize(loss, var_list=list(model_parameters.values()))

def split_data(X, y, validation_split):
    n_samples = X.shape[0]
    X_shuffled, y_shuffled = shuffle_data(X, y)
    n_validation_indices = np.floor(validation_split * n_samples).astype(int)
    X_val, y_val = X_shuffled[:n_validation_indices], y_shuffled[:n_validation_indices] if y is not None else None
    X_train, y_train = X_shuffled[n_validation_indices:], y_shuffled[n_validation_indices:] if y is not None else None
    return X_train, y_train, X_val, y_val

def shuffle_data(X, y):
    """Shuffle the data."""
    epoch_shuffled_indices = np.random.permutation(range(X.shape[0]))
    X_epoch = X[epoch_shuffled_indices]
    y_epoch = y[epoch_shuffled_indices] if y is not None else None
    return X_epoch, y_epoch

def create_mini_batch_data(X, y, mini_batch_indices):
    """Return a mini batch of the data."""
    X_batch = X[slice(*mini_batch_indices)]
    y_batch = y[slice(*mini_batch_indices)] if y is not None else None
    return X_batch, y_batch
                        
def mini_batch_indices_generator(n_samples, batch_size):
    """A generator of the mini batch indices based on the 
    number of samples and the batch size."""
    start_index = 0
    end_index = batch_size
    while start_index < n_samples:
        yield start_index, end_index
        start_index += batch_size
        if end_index + batch_size <= n_samples:
            end_index += batch_size
        else:
            end_index += n_samples % batch_size


class BaseGAN:
    """Base class for GANs and CGANs."""  

    def __init__(self, n_Z_features,
                       discriminator_hidden_layers, 
                       generator_hidden_layers, 
                       discriminator_optimizer=OPTIMIZER,  
                       discriminator_weights_initilization_choice='xavier',
                       discriminator_bias_initilization_choice='zeros',
                       generator_optimizer=OPTIMIZER, 
                       generator_weights_initilization_choice='xavier',
                       generator_bias_initilization_choice='zeros'):
        self.n_Z_features = n_Z_features
        self.discriminator_hidden__layers = discriminator_hidden_layers
        self.generator_hidden_layers = generator_hidden_layers
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_weights_initilization_choice = discriminator_weights_initilization_choice
        self.discriminator_bias_initilization_choice = discriminator_bias_initilization_choice
        self.generator_weights_initilization_choice = generator_weights_initilization_choice
        self.generator_bias_initilization_choice = generator_bias_initilization_choice

    def _initialize_training_parameters(self, X, y, batch_size):
        """Private method that initializes the GAN training parameters."""
        self.n_X_features = X.shape[1]
        self.n_y_features = y.shape[1] if y is not None else 0

        self.discriminator_layers = [(self.n_X_features + self.n_y_features, None)] + self.discriminator_hidden__layers + [(1, None)]
        self.generator_layers = [(self.n_Z_features + self.n_y_features, None)] + self.discriminator_hidden__layers + [(self.n_X_features, None)]

        self.y_placeholder = tf.placeholder(tf.float32, [None, self.n_y_features]) if y is not None else None
        self.X_placeholder, self.discriminator_parameters = initialize_model(self.discriminator_layers, self.n_y_features, self.discriminator_weights_initilization_choice, self.discriminator_bias_initilization_choice)
        self.Z_placeholder, self.generator_parameters = initialize_model(self.generator_layers, self.n_y_features, self.generator_weights_initilization_choice, self.generator_bias_initilization_choice)
        
        generator_logits = output_logits_tensor(bind_columns(self.Z_placeholder, self.y_placeholder), self.generator_layers, self.generator_parameters)
        discriminator_logits_real = output_logits_tensor(bind_columns(self.X_placeholder, self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        discriminator_logits_generated = output_logits_tensor(bind_columns(tf.nn.sigmoid(generator_logits), self.y_placeholder), self.discriminator_layers, self.discriminator_parameters)
        
        self.discriminator_loss_mixed_data = cross_entropy(discriminator_logits_real, True) + cross_entropy(discriminator_logits_generated, False)
        self.discriminator_loss_generated_data = cross_entropy(discriminator_logits_generated, True)

        self.discriminator_optimization = define_optimization(self.discriminator_optimizer, self.discriminator_loss_mixed_data, self.discriminator_parameters)
        self.generator_optimization = define_optimization(self.generator_optimizer, self.discriminator_loss_generated_data, self.generator_parameters)

        self.accuracy_mixed_data = 0.5 * accuracy(discriminator_logits_real, True) + 0.5 * accuracy(discriminator_logits_generated, False)
        self.accuracy_generated_data = accuracy(discriminator_logits_generated, True)

        self.discriminator_placeholders = [placeholder for placeholder in [self.X_placeholder, self.Z_placeholder, self.y_placeholder] if placeholder is not None]
        self.generator_placeholders = [placeholder for placeholder in [self.Z_placeholder, self.y_placeholder] if placeholder is not None]

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

    def _run_epoch_task(self, X, y, batch_size, task, placeholders):
        """Private method that returns the loss function value for an 
        epoch of training."""
        is_tensor = isinstance(task, tf.Tensor)
        n_samples = X.shape[0]
        X_epoch, y_epoch = shuffle_data(X, y)
        mini_batch_indices = mini_batch_indices_generator(n_samples, batch_size)
        n_batches = n_samples // batch_size
        if is_tensor:
            task_total = 0
        for batch_index in range(n_batches):
            mb_indices = next(mini_batch_indices)
            adjusted_batch_size = mb_indices[1] - mb_indices[0]
            X_batch, y_batch = create_mini_batch_data(X_epoch, y_epoch, mb_indices)
            feed_dict = {self.X_placeholder: X_batch, self.Z_placeholder: sample_Z(adjusted_batch_size, self.n_Z_features), self.y_placeholder: y_batch}
            feed_dict = {placeholder: data for placeholder, data in feed_dict.items() if placeholder in placeholders}
            task_mb = self.sess.run(task, feed_dict=feed_dict)
            if is_tensor:
                task_total += task_mb * adjusted_batch_size
        if is_tensor:
            return task_total / n_samples

    def _logging_info(self, X, y, X_train, y_train, X_val, y_val, options, epoch, batch_size, logging_steps, **kwargs):
        if epoch % logging_steps == 0:
            if 'print_accuracy' in options:
                accuracy_mixed_training_data = self._run_epoch_task(X_train, y_train, batch_size, self.accuracy_mixed_data, self.discriminator_placeholders)
                accuracy_generated_data = self._run_epoch_task(X_train, y_train, batch_size, self.accuracy_generated_data, self.generator_placeholders)
                accuracy_mixed_validation_data = None
                if X_val.size > 0:
                    accuracy_mixed_validation_data = self._run_epoch_task(X_val, y_val, batch_size, self.accuracy_mixed_data, self.discriminator_placeholders)
                accuracy_types = {'mixed training': accuracy_mixed_training_data, 'mixed validation': accuracy_mixed_validation_data, 'generated': accuracy_generated_data}
                accuracy_types = {key:value for key, value in accuracy_types.items() if value is not None}
                msg = 'Epoch: {}'
                for key in accuracy_types.keys():
                    msg += '\nDiscriminator accuracy on ' + key + ' data: {:.3f}'
                print((msg + '\n').format(epoch, *accuracy_types.values()))
        
            if 'plot_images' in options:
                n_samples = kwargs['n_samples']
                if self.n_y_features == 0:
                    X_generated = self.generate_samples(n_samples)
                else:
                    X_generated = self.generate_samples(n_samples, kwargs['class_label'])
                plt.rcParams['figure.figsize'] = (50, 50)
                fig, ax = plt.subplots(1, n_samples)
                img_dim = int(sqrt(self.n_X_features))
                X_img = X_generated.reshape(n_samples, img_dim, -1)
                for ind in range(n_samples):    
                    ax[ind].imshow(X_img[ind], cmap='gray_r')
                    ax[ind].axis('off')
                plt.show()
                
            if 'plot2d_data' in options:
                n_samples = kwargs['n_samples']
                feature_index1, feature_index2 = kwargs['feature_index1'], kwargs['feature_index2']
                if self.n_y_features == 0:
                    X_generated = self.generate_samples(n_samples)
                else:
                    class_label = kwargs['class_label']
                    X_generated = self.generate_samples(n_samples, class_label)
                X1 = np.concatenate([X_train[:, feature_index1], X_generated[:, feature_index1]])
                X2 = np.concatenate([X_train[:, feature_index2], X_generated[:, feature_index2]])
                pl.clf()
                if self.n_y_features == 0:
                    pl.scatter(X1, X2, c=X_train.shape[0] * ['blue'] + n_samples * ['red'])
                else:
                    random.seed(a=1)
                    random_colors = random.choices(list(mcolors.cnames.keys()), k=self.n_y_features + 1 if self.n_y_features > 1 else 3)
                    generated_color, *class_colors = random_colors
                    generated_color = np.array(n_samples * [generated_color])
                    class_colors = np.array(class_colors)[y_train.argmax(axis=1) if self.n_y_features > 1 else y_train.reshape(-1)]
                    colors = np.concatenate([class_colors, generated_color])
                    pl.scatter(X1, X2, c=colors)
                pl.xlabel('Feature 1'), pl.ylabel('Feature 2')
                display.display(pl.gcf())
                display.clear_output(wait=True)


class GAN(BaseGAN):
    """
    Parameters
    ----------
    discriminator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the discriminator's corresponding layer. The 
        number of neurons and the activation function for the first 
        layer should be equal to (X.shape[1], None) while the number 
        of neurons and the activation function for the last layer should 
        be equal to (1, None), where X is the input data.
    generator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the generators's corresponding layer. The 
        number of neurons and the activation function for the first 
        layer should be equal (Z.shape[1], None)while the number of neurons 
        and the activation function for the last layer should be equal to 
        (X.shape[1], None), where X is the input data.
    discriminator_optimizer: TensorFlow optimizer
        The optimizer for the discriminator.
    generator_optimizer: TensorFlow optimizer
        The optimizer for the generator.
    """

    def train(self, X, nb_epoch, batch_size, validation_split=0.0, discriminator_steps=1, logging_options=['print'], logging_steps=1, **kwargs):
        """Trains the GAN with X as the input data for nb_epoch number of epochs, 
        batch_size the size of the mini batch and discriminator_steps as the number 
        of discriminator gradient updates for each generator gradient update."""
        X_train, y_train, X_val, y_val = split_data(X, None, validation_split)
        super()._initialize_training_parameters(X_train, y_train, batch_size)
        for epoch in range(nb_epoch):
            X_train, y_train = shuffle_data(X_train, y_train)
            for _ in range(discriminator_steps):
                self._run_epoch_task(X_train, y_train, batch_size, self.discriminator_optimization, self.discriminator_placeholders)
            self._run_epoch_task(X_train, y_train, batch_size, self.generator_optimization, self.generator_placeholders)
            self._logging_info(X, None, X_train, y_train, X_val, y_val, logging_options, epoch, batch_size, logging_steps, **kwargs)
        return self
            
    def generate_samples(self, n_samples):
        """Generates n_samples from the generator."""
        input_tensor = sample_Z(n_samples, self.n_Z_features)
        logits = output_logits_tensor(input_tensor, self.generator_layers, self.generator_parameters)
        generated_samples = self.sess.run(tf.nn.sigmoid(logits))
        return generated_samples


class CGAN(BaseGAN):
    """
    Parameters
    ----------
    discriminator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the discriminator's corresponding layer. The number 
        of neurons and the activation function for the first layer should 
        be equal to (X.shape[1] + n_y_features, None) while the number of 
        neurons and the activation function for the last layer should be 
        equal to (1, None), where X is the input data and n_y_features is 
        the number of y matrix features.
    generator_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the generators's corresponding layer. The number 
        of neurons and the activation function for the first layer should 
        be equal to (Z.shape[1] + n_y_features, None) while the number of 
        neurons and the activation function for the last layer should be 
        equal to (X.shape[1], None), where X is the input data and n_y_features 
        is the number of y matrix features.
    discriminator_optimizer: TensorFlow optimizer
        The optimizer for the discriminator.
    generator_optimizer: TensorFlow optimizer
        The optimizer for the generator.
    """

    def train(self, X, y, nb_epoch, batch_size, validation_split=0.0, discriminator_steps=1, logging_options=['print'], logging_steps=1, **kwargs):
        """Trains the Conditional GAN with X as the input data, y the one-hot
        encoded class labels for nb_epoch number of epochs, batch_size the size 
        of the mini batch, discriminator_steps as the number of discriminator 
        gradient updates for each generator gradient update."""
        X_train, y_train, X_val, y_val = split_data(X, y, validation_split)
        super()._initialize_training_parameters(X_train, y_train, batch_size)
        for epoch in range(nb_epoch):
            X_train, y_train = shuffle_data(X_train, y_train)
            for _ in range(discriminator_steps):
                self._run_epoch_task(X_train, y_train, batch_size, self.discriminator_optimization, self.discriminator_placeholders)
            self._run_epoch_task(X_train, y_train, batch_size, self.generator_optimization, self.generator_placeholders)
            self._logging_info(X, y, X_train, y_train, X_val, y_val, logging_options, epoch, batch_size, logging_steps, **kwargs)
        return self

    def generate_samples(self, n_samples, class_label):
        """Generates n_samples number from the generator 
        conditioned on the class_label."""
        input_tensor = np.concatenate([sample_Z(n_samples, self.n_Z_features), sample_y(n_samples, self.n_y_features, class_label)], axis=1)
        logits = output_logits_tensor(input_tensor, self.generator_layers, self.generator_parameters)
        generated_samples = self.sess.run(tf.nn.sigmoid(logits))
        return generated_samples
    
    def _logging_info(self, X, y, X_train, y_train, X_val, y_val, options, epoch, batch_size, logging_steps, **kwargs):
        super()._logging_info(X, y, X_train, y_train, X_val, y_val, options, epoch, batch_size, logging_steps, **kwargs)
        if 'evaluate_oversampling' in options:
            n_samples = kwargs['n_samples']
            class_label = kwargs['class_label']
            if epoch == 0:
                self.clf = LogisticRegression()
                self.clf.fit(X_train, y_train.reshape(-1))
                self.unbalanced_data_auc = roc_auc_score(y_val, self.clf.predict_proba(X_val)[:, 1])
            if epoch % logging_steps == 0:
                X_generated = self.generate_samples(n_samples, class_label)
                X_over = np.concatenate([X_train, X_generated])
                y_over = np.concatenate([y_train.reshape(-1), np.array(n_samples * [class_label])])
                self.clf.fit(X_over, y_over)
                self.balanced_data_auc =  roc_auc_score(y_val, self.clf.predict_proba(X_val)[:, 1])
                print('Epoch: {}\nUnbalanced data validation AUC: {:.3f}\nBalanced data validation AUC: {:.3f}\n'.format(epoch, self.unbalanced_data_auc, self.balanced_data_auc))

                