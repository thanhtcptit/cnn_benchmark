import importlib
import tensorpack as tp
import tensorflow as tf

from src.models.base import BaseClassifier


@BaseClassifier.register('external')
class ExternalModel(BaseClassifier):
    def __init__(self, image_size, num_labels, model_def, embedding_dim,
                 keep_pr, l2_regular, optimizer, learning_rate):
        super(ExternalModel, self).__init__(
            image_size, num_labels, embedding_dim, keep_pr, l2_regular,
            optimizer, learning_rate)
        self.model_def = model_def
        self.network = importlib.import_module(model_def)

    def inference(self, image):
        is_training = tp.get_current_tower_context().is_training
        prelogits, _ = self.network.inference(
            image, keep_probability=self.keep_pr,
            phase_train=is_training,
            bottleneck_layer_size=self.embedding_dim,
            weight_decay=self.l2_regular)
        return prelogits
