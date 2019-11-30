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
    callbacks = []

    callback_module = importlib.import_module('src.callbacks')
    for name, params in callback_params.items():
        params = dict(params)
        callback_class = getattr(callback_module, name)
        callback_inst = callback_class(**params)
        callbacks.append(callback_inst)
    return callbacks


class MLflowLogging(Callback):
    def __init__(self, exp_name, stats, params, artifacts, trigger_every=1):
        if not isinstance(stats, list):
            stats = [stats]
        self._stats = stats
        self.artifacts = artifacts

        self._trigger_every = trigger_every

        env_cfg = load_json('protected_asset/env.json')
        tracking_server_cfg = env_cfg['tracking_server']
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
            tracking_server_cfg['gcloud_storage_key']
        mlflow.set_tracking_uri(tracking_server_cfg['tracking_server_addr'])
        mlflow.set_experiment(exp_name)
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
    def __init__(self, checkpoint_path, fix_weight=False):
        self._checkpoint_path = checkpoint_path
        self._fix_weight = fix_weight
        self._saver = None

    def _setup_graph(self):
        all_trainable_vars = tf.get_collection_ref(
            tf.GraphKeys.TRAINABLE_VARIABLES)
        user_embs_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='user_embs')
        encoder_vars = []
        for var in all_trainable_vars:
            if var not in user_embs_var and 'EMA' not in var:
                encoder_vars.append(var)
        if self._fix_weight:
            for var in encoder_vars:
                all_trainable_vars.remove(var)

        self._saver = tf.train.Saver(encoder_vars)

    def _before_train(self):
        self._saver.restore(self.trainer.sess, self._checkpoint_path)
