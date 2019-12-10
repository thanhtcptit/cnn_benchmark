import tensorflow as tf
import tensorpack as tp

from nsds.common.registrable import Registrable
from tensorpack.tfutils.summary import add_moving_summary
from tensorflow.python.framework.tensor_spec import TensorSpec


class BaseClassifier(tp.ModelDesc, Registrable):
    """Base class for image classification
    Any subclass will need to implement :meth: inference()
    """

    def __init__(self, image_size, num_labels, embedding_dim,
                 keep_pr, l2_regular, optimizer, learning_rate):
        self.image_size = image_size
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.keep_pr = keep_pr
        self.l2_regular = l2_regular
        self._optimizer = optimizer
        self.learning_rate = learning_rate

        tf.reset_default_graph()

    def inputs(self):
        return [TensorSpec([None, self.image_size, self.image_size, 3],
                           tf.float32, name='input'),
                TensorSpec((None,), tf.int32, name='label')]

    def inference(self, *args, **kwargs):
        raise NotImplementedError

    def build_graph(self, image, label):
        """
        Takes inputs tensors that matches what you’ve defined in inputs().
        Need to return `total_cost` Tensor if using `SingleCostTrainer`
        such as `SyncMultiGPUTrainerParameterServer`
        """
        prelogits = self.inference(image)
        embeddings = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10,
                                        name='embeddings')
        logits = tp.FullyConnected(
            'Logits', prelogits, self.num_labels,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                self.l2_regular))

        # Build loss func
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='loss')
        add_moving_summary(loss)

        # Calculate prediction
        prediction = tf.nn.softmax(logits, name='prediction')
        top3_conf, top3_class = tf.nn.top_k(prediction, k=3)
        top3_conf = tf.identity(top3_conf, name='top3_conf')
        top3_class = tf.identity(top3_class, name='top3_class')

        def topk_incorrect_preds(logits, label, k, name):
            return tf.cast(tf.logical_not(tf.nn.in_top_k(predictions=logits,
                                                         targets=label, k=k)),
                           dtype=tf.float32, name=name)

        top1_incorrect_preds = topk_incorrect_preds(logits, label, k=1,
                                                    name='top1-error')
        add_moving_summary(tf.reduce_mean(top1_incorrect_preds,
                                          name='train-top1-error'))
        top3_incorrect_preds = topk_incorrect_preds(logits, label, k=3,
                                                    name='top3-error')
        add_moving_summary(tf.reduce_mean(top3_incorrect_preds,
                                          name='train-top3-error'))
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate,
                             trainable=False)
        add_moving_summary(lr)
        if self._optimizer == 'ADAM':
            return tf.train.AdamOptimizer(
                learning_rate=lr, epsilon=0.1)
        elif self._optimizer == 'ADAGRAD':
            return tf.train.AdagradOptimizer(learning_rate=lr)
        elif self._optimizer == 'RMSPROP':
            return tf.train.RMSPropOptimizer(
                learning_rate=lr, decay=0.9, momentum=0.9, epsilon=1.0)
        elif self._optimizer == 'MOM':
            return tf.train.MomentumOptimizer(
                learning_rate=lr, momentum=0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
