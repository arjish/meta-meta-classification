import numpy as np
import tensorflow as tf


class MAML:
    def __init__(self, data_source, kclass, kshot, kquery, train_lr=1e-4, meta_lr=1e-3, scope='MAML'):
        """

        :param data_source:
        :param kclass:
        :param kshot:
        :param kquery:
        :param train_lr:
        :param meta_lr:
        :param scope:
        """
        self.data_source = data_source
        self.kclass = kclass
        self.scope = scope

        if data_source == 'omniglot':
            self.image_size = 28
            self.num_filters = 64
            self.num_channels = 1
            self.maxpooling = False
            self.gradclipping = False
        else:
            self.image_size = 84
            self.num_filters = 32
            self.num_channels = 3
            self.maxpooling = True
            self.gradclipping = True

        # tackling class imbalance during training
        self.w_negative = float((kshot + kquery) / kquery)
        self.w_positive = float((kshot + kquery) / kshot)

        self.meta_lr = meta_lr
        self.train_lr = train_lr

        print('img shape:', self.image_size, self.image_size,
            self.num_channels, 'meta-lr:', meta_lr, 'train-lr:', train_lr)

    def build(self, support_xb, support_yb, query_xb,
              query_yb, K, meta_batchsz, mode='train'):
        """

        :param support_xb:   [b, setsz, 84*84*3]
        :param support_yb:   [b, setsz, kclass]
        :param query_xb:     [b, querysz, 84*84*3]
        :param query_yb:     [b, querysz, kclass]
        :param K:           train update steps
        :param meta_batchsz:tasks number
        :param mode:        train/eval/test, for training, we build train&eval network meanwhile.
        :return:
        """
        # create or reuse network variable, not including batch_norm variable, therefore we need extra reuse mechnism
        # to reuse batch_norm variables.
        self.weights = self.conv_weights()
        # TODO: meta-test is sort of test stage.
        training = True if mode is 'train' else False
        if training:
            keep_prob_in = tf.constant(0.8, dtype=tf.float32)
        else:
            keep_prob_in = tf.constant(0.6, dtype=tf.float32)


        def meta_task(input):
            """
            map_fn only support one parameters, so we need to unpack from tuple.
            :param support_x:   [setsz, 84*84*3]
            :param support_y:   [setsz, kclass]
            :param query_x:     [querysz, 84*84*3]
            :param query_y:     [querysz, kclass]
            :param training:    training or not, for batch_norm
            :return:
            """
            support_x, support_y, query_x, query_y = input
            # support_x = tf.nn.dropout(support_x, keep_prob_in)
            # query_x = tf.nn.dropout(query_x, keep_prob_in)
            # to record the op in t update step.
            query_preds, query_losses, query_accs = [], [], []

            support_pred = self.forward(support_x, self.weights, training)

            # Minimize error using weighted cross entropy ::
            class_weight = tf.constant([[self.w_negative, self.w_positive]])
            weight_per_label = tf.transpose(
                tf.matmul(support_y, tf.transpose(class_weight)))  # [1, n_problems_train]

            # This is the weight for each datapoint, depending on its label
            support_loss = tf.multiply(weight_per_label,
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=support_y,
                    logits=support_pred))  # shape [1, n_problems_train]

            # support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
            support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, axis=1), axis=1),
                tf.argmax(support_y, axis=1))
            # compute gradients
            grads = tf.gradients(support_loss, list(self.weights.values()))
            # grad and variable dict
            gvs = dict(zip(self.weights.keys(), grads))

            # theta_pi = theta - alpha * grads
            fast_weights = dict(zip(self.weights.keys(),
                [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
            # use theta_pi to forward meta-test
            query_pred = self.forward(query_x, fast_weights, training)
            # meta-test loss
            query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
            # record T0 pred and loss for meta-test
            query_preds.append(query_pred)
            query_losses.append(query_loss)

            # continue to build T1-TK steps graph
            for _ in range(1, K):
                # T_k loss on meta-train
                # we need meta-train loss to fine-tune the task and meta-test loss to update theta
                support_pred = self.forward(support_x, fast_weights, training)
                support_loss = tf.multiply(weight_per_label,
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=support_y, logits=support_pred))
                # support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
                support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, axis=1), axis=1),
                    tf.argmax(support_y, axis=1))

                # compute gradients
                grads = tf.gradients(support_loss, list(fast_weights.values()))
                # compose grad and variable dict
                gvs = dict(zip(fast_weights.keys(), grads))
                # update theta_pi according to varibles
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key]
                                                              for key in fast_weights.keys()]))
                # forward on theta_pi
                query_pred = self.forward(query_x, fast_weights, training)
                # we need accumulate all meta-test losses to update theta
                query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
                query_preds.append(query_pred)
                query_losses.append(query_loss)

            # compute every steps' accuracy on query set
            for i in range(K):
                query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], axis=1), axis=1),
                    tf.argmax(query_y, axis=1)))
            # we just use the first step support op: support_pred & support_loss, but igonre these support op
            # at step 1:K-1.
            # however, we return all pred&loss&acc op at each time steps.
            result = [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]

            return result

        # return: [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
        out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
        result = tf.map_fn(meta_task, elems=(support_xb, support_yb, query_xb, query_yb),
            dtype=out_dtype, parallel_iterations=meta_batchsz, name='map_fn')
        support_pred_tasks, support_loss_tasks, support_acc_tasks, \
        query_preds_tasks, query_losses_tasks, query_accs_tasks = result

        if mode is 'train':
            # average loss
            self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
            # [avgloss_t1, avgloss_t2, ..., avgloss_K]
            self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                for j in range(K)]
            # average accuracy
            self.support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
            # average accuracies
            self.query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
                                            for j in range(K)]
            self.query_preds = query_preds_tasks[-1]
            self.query_preds_probs = tf.nn.softmax(query_preds_tasks[-1], axis=-1)

            self.support_preds = support_pred_tasks[-1]
            self.support_preds_probs = tf.nn.softmax(support_pred_tasks[-1], axis=-1)

            # # add batch_norm ops before meta_op
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            # 	# TODO: the update_ops must be put before tf.train.AdamOptimizer,
            # 	# otherwise it throws Not in same Frame Error.
            # 	meta_loss = tf.identity(self.query_losses[-1])

            # meta-train optim
            optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
            # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
            gvs = optimizer.compute_gradients(self.query_losses[-1])
            # meta-train grads clipping
            if self.gradclipping:
                gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
            # update theta
            self.meta_op = optimizer.apply_gradients(gvs)


        elif mode is 'testEach':
            # average accuracy
            self.test_support_acc = support_acc_tasks
            # average accuracies
            self.support_preds = support_pred_tasks
            self.test_query_accs = query_accs_tasks[-1]
            self.query_preds = query_preds_tasks[-1]
            self.query_preds_probs = tf.nn.softmax(query_preds_tasks[-1], axis=-1)
        else:  # test & eval
            # average loss
            self.test_support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
            # [avgloss_t1, avgloss_t2, ..., avgloss_K]
            self.test_query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                     for j in range(K)]
            # average accuracy
            self.test_support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
            # average accuracies
            self.test_query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
                                                 for j in range(K)]
            self.query_preds_probs = tf.nn.softmax(query_preds_tasks[-1], axis=-1)


    def conv_weights(self):
        weights = {}

        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        fc_initializer = tf.contrib.layers.xavier_initializer()
        k = 3

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights['conv1'] = tf.get_variable('conv1w', [k, k, self.num_channels, self.num_filters], initializer=conv_initializer)
            weights['b1'] = tf.get_variable('conv1b', initializer=tf.zeros([self.num_filters]))
            weights['conv2'] = tf.get_variable('conv2w', [k, k, self.num_filters, self.num_filters], initializer=conv_initializer)
            weights['b2'] = tf.get_variable('conv2b', initializer=tf.zeros([self.num_filters]))
            weights['conv3'] = tf.get_variable('conv3w', [k, k, self.num_filters, self.num_filters], initializer=conv_initializer)
            weights['b3'] = tf.get_variable('conv3b', initializer=tf.zeros([self.num_filters]))
            weights['conv4'] = tf.get_variable('conv4w', [k, k, self.num_filters, self.num_filters], initializer=conv_initializer)
            weights['b4'] = tf.get_variable('conv4b', initializer=tf.zeros([self.num_filters]))

            if self.maxpooling:
                # assumes max pooling
                weights['w5'] = tf.get_variable('fc1w', [self.num_filters * 5 * 5, self.kclass],
                    initializer=fc_initializer)
                weights['b5'] = tf.get_variable('fc1b', initializer=tf.zeros([self.kclass]))
            else:
                weights['w5'] = tf.get_variable('fc1w', [self.num_filters, self.kclass],
                    initializer=fc_initializer)
                weights['b5'] = tf.get_variable('fc1b', initializer=tf.zeros([self.kclass]))

            return weights

    def conv_block(self, x, weight, bias, scope, training):
        """
        build a block with conv2d->batch_norm->pooling
        :param x:
        :param weight:
        :param bias:
        :param scope:
        :param training:
        :return:
        """
        # conv
        stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]

        if self.maxpooling:
            x = tf.nn.conv2d(x, weight, no_stride, 'SAME', name=scope + '_conv2d') + bias
        else:
            x = tf.nn.conv2d(x, weight, stride, 'SAME', name=scope + '_conv2d') + bias
        # batch norm, activation_fn=tf.nn.relu,
        # NOTICE: must have tf.layers.batch_normalization

        # CANNOT do batch-norm if there's an uneven distribution of images per class
        # with tf.variable_scope(self.scope):
        #     # train is set to True ALWAYS, please refer to https://github.com/cbfinn/maml/issues/9
        #     # when FLAGS.train=True, we still need to build evaluation network
        #     x = tf.layers.batch_normalization(x, training=training, name=scope + '_bn', reuse=tf.AUTO_REUSE)

        # relu
        x = tf.nn.relu(x, name=scope + '_relu')

        # pooling
        if self.maxpooling:
            x = tf.nn.max_pool(x, stride, stride, 'VALID', name=scope + '_pool')

        return x

    def forward(self, x, weights, training):
        """


        :param x:
        :param weights:
        :param training:
        :return:
        """

        if training:
            keep_prob_hidden = tf.constant(1.0, dtype=tf.float32)
        else:
            keep_prob_hidden = tf.constant(1.0, dtype=tf.float32)

        # [b, 84, 84, 3]
        x = tf.reshape(x, [-1, self.image_size, self.image_size, self.num_channels], name='reshape1')

        hidden1 = self.conv_block(x, weights['conv1'], weights['b1'], 'conv0', training)
        # hidden1 = tf.nn.dropout(hidden1, keep_prob_hidden)

        hidden2 = self.conv_block(hidden1, weights['conv2'], weights['b2'], 'conv1', training)
        # hidden2 = tf.nn.dropout(hidden2, keep_prob_hidden)

        hidden3 = self.conv_block(hidden2, weights['conv3'], weights['b3'], 'conv2', training)
        # hidden3 = tf.nn.dropout(hidden3, keep_prob_hidden)

        hidden4 = self.conv_block(hidden3, weights['conv4'], weights['b4'], 'conv3', training)
        # hidden4 = tf.nn.dropout(hidden4, keep_prob_hidden)

        # get_shape is static shape, (5, 5, 5, self.num_filters)
        # print('flatten:', hidden4.get_shape())
        # flatten layer
        if self.maxpooling:
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])], name='reshape2')
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        output = tf.add(tf.matmul(hidden4, weights['w5']), weights['b5'], name='fc1')

        return output
