import tensorflow as tf

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)

class MOE:
    def __init__(self, nway, kshot, kquery, n_models, feature_size,
                 meta_batchsz, lr=1e-3):
        """
        :param nway:
        :param kshot:
        :param kquery:
        :param n_models:
        :param feature_size:
        :param meta_batchsz:
        :param lr:
        """
        self.nway = nway
        self.kshot = kshot
        self.kquery = kquery
        self.n_models = n_models
        self.meta_batchsz = meta_batchsz
        self.kclass = 2

        self.lr = lr
        self.hidden_size = 256

        self.n_images_query = kquery * nway
        ##To make balanced positive examples:
        # self.n_images_query = 2 * kquery * (nway - 1)

        self.support_f = tf.placeholder(tf.float32,
            shape=[meta_batchsz, feature_size], name='support_f')
        self.query_preds = tf.placeholder(tf.float32,
            shape=[n_models, meta_batchsz, self.n_images_query, self.kclass], name='query_preds')
        self.query_y = tf.placeholder(tf.int32,
            shape=[meta_batchsz, self.n_images_query], name='query_y')

        self.keep_prob_in = tf.placeholder(tf.float32, name='keep_prob_in')
        self.keep_prob_hidden = tf.placeholder(tf.float32, name='keep_prob_hidden')
        self.build_model()

        print('train-lr:', lr)

    def build_model(self):
        query_y = tf.one_hot(self.query_y, self.kclass)

        qp = tf.reshape(tf.transpose(self.query_preds, [1,2, 0,3]),
            [self.meta_batchsz, self.n_images_query, -1])

        f = tf.tile(tf.expand_dims(self.support_f, 1), [1, self.n_images_query, 1])

        input = tf.concat([f, qp], axis =2)

        dropout_input = tf.nn.dropout(input, self.keep_prob_in)
        with tf.variable_scope('meta_classifier'):
            hidden = tf.layers.dense(
                inputs=dropout_input,
                units=self.hidden_size,  # number of hidden units
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0, 1),  # biases
                name='hidden'
            )
        
        dropout_hidden = tf.nn.dropout(hidden, self.keep_prob_hidden)
        with tf.variable_scope('meta_classifier'):
            hidden_2 = tf.layers.dense(
                inputs=dropout_hidden,
                units=self.hidden_size,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0, 1),  # biases
                name='hidden_2'
            )

        dropout_hidden_2 = tf.nn.dropout(hidden_2, self.keep_prob_hidden)
        with tf.variable_scope('meta_classifier'):
            query_preds_moe = tf.layers.dense(
                inputs=dropout_hidden_2,
                units=2,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0, 1),  # biases
                name='query_preds_moe'
            )

        self.query_preds_moe = query_preds_moe
        query_probs_moe = tf.nn.softmax(query_preds_moe, axis=-1)
        self.query_probs_moe = query_probs_moe

        ##Query ::
        # this is the weight for each datapoint, depending on its label
        query_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=query_y,
                logits=query_preds_moe)  # shape [meta_batchsz* n_examples]
        # average loss
        self.query_loss = tf.reduce_sum(query_loss) / self.meta_batchsz
        self.query_acc = tf.contrib.metrics.accuracy(tf.argmax(query_probs_moe, axis=-1),
            tf.argmax(query_y, axis=-1))

        # meta-train optim
        optimizer = tf.train.AdamOptimizer(self.lr, name='meta_optim')
        # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
        self.updateModel = optimizer.minimize(self.query_loss)


