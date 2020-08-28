import numpy as np
import sys
import torch
from torch import nn
from torch.nn import Module, init, functional as F


def xent(pred, label):
    return F.nll_loss(pred, label)

def conv_block(inp, channel_weight, bias_weight, activation=torch.relu):
    stride, no_stride=[1, 1, 2, 2], [1, 1, 1, 1]

    conv_output = F.conv2d(inp, channel_weight, bias=bias_weight, stride=no_stride, padding=len(channel_weight) // 2)
    normed = activation(F.batch_norm(conv_output))
    return normed


class MAML:
    def __init__(self, args, dim_input=1, dim_output=1, meta_test_num_inner_updates=5):
        """must call construct_model() after initializing MAML! """
        self.args = args
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = args.inner_update_lr
        self.meta_lr = args.meta_lr
        self.meta_test_num_inner_updates = meta_test_num_inner_updates
        self.loss_func = xent
        self.dim_hidden = args.num_filters
        self.forward = self.forward_conv
        self.construct_weights = self.construct_conv_weights
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

    def construct_model(self, prefix='maml'):
        # a: group of data for calculating inner gradient
        # b: group of data for evaluating modified weights and computing meta gradient
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:
            # outputbs[i] and lossesb[i] are the output and loss after i+1 inner gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            # number of loops in the inner training loop
            num_inner_updates = max(self.meta_test_num_inner_updates, self.args.num_inner_updates)
            outputbs = [[]] * num_inner_updates
            lossesb = [[]] * num_inner_updates
            accuraciesb = [[]] * num_inner_updates

            if 'weights' in dir(self):
                weights = self.weights
            else:
                # Define the weights - these should NOT be directly modified by the
                # inner training loop
                self.weights = weights = self.construct_weights()

            def task_inner_loop(inp):
                """
                    Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
                    Args:
                        inp: a tuple (inputa, inputb, labela, labelb), where inputa and labela are the inputs and
                            labels used for calculating inner loop gradients and inputa and labela are the inputs and
                            labels used for evaluating the model after inner updates.
                        reuse: reuse the model parameters or not. Hint: You can just pass its default value to the
                            forwawrd function
                    Returns:
                        task_output: a list of outputs, losses and accuracies at each inner update
                """
                inputa, inputb, labela, labelb = inp

                #############################
                #### YOUR CODE GOES HERE ####
                # perform num_inner_updates to get modified weights
                # modified weights should be used to evaluate performance
                # Note that at each inner update, always use inputa and labela for calculating gradients
                # and use inputb and labels for evaluating performance
                # HINT: you may wish to use tf.gradients()

                # output, loss, and accuracy of group a before performing inner gradientupdate
                task_outputa, task_lossa, task_accuracya = None, None, None
                # lists to keep track of outputs, losses, and accuracies of group b for each inner_update
                # where task_outputbs[i], task_lossesb[i], task_accuraciesb[i] are the output, loss, and accuracy
                # after i+1 inner gradient updates
                task_outputbs, task_lossesb, task_accuraciesb = [], [], []
                #############################

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, task_accuracya, task_accuraciesb]

                return task_output

            # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
            unused = task_inner_loop((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)
            out_dtype = [tf.float32, [tf.float32] * num_inner_updates, tf.float32, [tf.float32] * num_inner_updates]
            out_dtype.extend([tf.float32, [tf.float32] * num_inner_updates])
            result = tf.map_fn(task_inner_loop, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        ## Performance & Optimization
        self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in
                                              range(num_inner_updates)]
        # after the map_fn
        self.outputas, self.outputbs = outputas, outputbs
        self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
        self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                      for j in range(num_inner_updates)]

        if FLAGS.meta_train_iterations > 0:
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_inner_updates - 1])
            self.metatrain_op = optimizer.apply_gradients(gvs)

        ## Summaries
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        tf.summary.scalar(prefix + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_inner_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])

    def construct_conv_weights(self):
        weights = {}

        dtype = torch.float32
        conv_initializer = init.xavier_uniform_

        k = 3

        weights['conv1'] = conv_initializer(torch.rand([k, k, self.channels, self.dim_hidden], dtype=dtype, requires_grad=True))
        weights['b1'] = torch.zeros([self.dim_hidden], dtype=dtype, requires_grad=True)
        weights['conv2'] = conv_initializer(torch.rand([k, k, self.channels, self.dim_hidden], dtype=dtype, requires_grad=True))
        weights['b2'] = torch.zeros([self.dim_hidden], dtype=dtype, requires_grad=True)
        weights['conv3'] = conv_initializer(torch.rand([k, k, self.channels, self.dim_hidden], dtype=dtype, requires_grad=True))
        weights['b3'] = torch.zeros([self.dim_hidden], dtype=dtype, requires_grad=True)
        weights['conv4'] = conv_initializer(torch.rand([k, k, self.channels, self.dim_hidden], dtype=dtype, requires_grad=True))
        weights['b4'] = torch.zeros([self.dim_hidden], dtype=dtype, requires_grad=True)

        weights['w5'] = torch.randn([self.dim_hidden, self.dim_output], dtype=dtype, requires_grad=True)
        weights['b5'] = torch.zeros([self.dim_output], dtype=dtype, requires_grad=True)

        return weights

    def forward_conv(self, inp, weights):
        channels = self.channels
        inp = torch.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'])
        hidden2= conv_block(hidden1, weights['conv2'], weights['b2'])
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'])
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'])
        hidden4 = torch.sum(hidden4, [1, 2])

        return torch.mm(hidden4, weights['w5']) + weights['b5']
