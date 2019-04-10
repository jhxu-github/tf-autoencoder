import  numpy as np
import  tensorflow as tf
import  pandas as pd

class Autoencoder():
    def __init__(self,n_hidden_1,n_hidden_2,n_input,learning_rate):
        #init   var
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2, = n_hidden_2
        self.n_input = n_input
        self.learning_rate = learning_rate
        self.weights ,self.biases = self._initialize_weights()
        self.x = tf.placeholder("float",[None,self.n_input])
        # coder op
        self.encoder_op = self.encoder(self.x)
        self.decoder_op = self.decoder(self.encoder_op)
        #cost and optimizer
        self.cost = tf.reduce_mean(tf.pow(self.x - self.decoder_op,2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        ###define weights and biases
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input,self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2,self.n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_input]))
        }

        return weights, biases

    def encoder(self,X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X,self.weights['encoder_h1']),self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))

        return layer_2

    def decoder(self,X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['decoder_h1']), self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))

        return layer_2

    def calc_total_cost(self,X):
        return  self.sess.run(self.cost,feed_dict={self.x: X})

    def partial_fit(self,X):
        cost, opt = self.sess.run((self.cost,self.optimizer),feed_dict={self.x: X})
        return cost
    def transform(self,X):
        return self.sess.run(self.encoder_op,feed_dict={self.x: x})
    def reconstruct(self,X):
        return self.sess.run(slef.decoder_op,feed_dict={self.x: X})