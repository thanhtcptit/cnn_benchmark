import os
import inspect
import mlflow
import importlib
import tensorflow as tf

from tensorpack.callbacks import Callback

from src.utils.common import load_json
from src.callbacks.hyperparam_setter import *


def parse_callback_params(callback_params):
    callback_params = dict(callback_params)
    preorder_callbacks = []
    callbacks = []

    callback_module = importlib.import_module('src.callbacks')
    for name, params in callback_params.items():
        params = dict(params)
        priority = params.pop('priority')
        callback_class = getattr(callback_module, name)
        callback_inst = callback_class(**params)
        preorder_callbacks.append([priority, callback_inst])
    preorder_callbacks.sort(key=lambda x: x[0],
                            reverse=True)
    for _, callback in preorder_callbacks:
        callbacks.append(callback)
    return callbacks


class MLflowLogging(Callback):
    def __init__(self, experiment_name, stats, params, artifacts=None,
                 trigger_every=1, tracking_uri="", auth_key=None):
        if not isinstance(stats, list):
            stats = [stats]
        self._stats = stats

        if not isinstance(params, dict):
            params = dict(params)

        if artifacts is None:
            artifacts = []
        elif not isinstance(artifacts, list):
            artifacts = [artifacts]
        self.artifacts = artifacts

        self._trigger_every = trigger_every

        if auth_key:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_key
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run_obj = mlflow.start_run()

        mlflow.log_params(params)

    def _trigger(self):
        if self.epoch_num % self._trigger_every == 0:
            self._log_metrics()

    def _after_train(self):
        for artifact in self.artifacts:
            if not os.path.exists(artifact):
                print(f"Artifact '{artifact}' doesn't exsist")
                continue
            mlflow.log_artifact(artifact)

    def _log_metrics(self):
        m = self.trainer.monitors
        metrics = {k: m.get_latest(k) for k in self._stats}
        mlflow.log_metrics(metrics)


class LoadPretrainWeight(Callback):
    def __init__(self, checkpoint_path, scope=None):
        self._checkpoint_path = checkpoint_path
        self._scope = scope
        self._saver = None

    def _setup_graph(self):
        if self._scope:
            load_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)
        else:
            load_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES)

        self._saver = tf.train.Saver(load_weights)

    def _before_train(self):
        self._saver.restore(self.trainer.sess, self._checkpoint_path)
