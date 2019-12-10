import os
import shutil
import random
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorpack.utils import logger
from tensorpack.utils import fix_rng_seed
from tensorpack.callbacks import ModelSaver, InferenceRunner, MinSaver, \
    ClassificationError, ScalarStats
from tensorpack.train import AutoResumeTrainConfig, \
    SyncMultiGPUTrainerParameterServer, launch_train_with_config

from src.models.base import BaseClassifier
from src.data.dataflow import get_image_dataflow
from src.callbacks.utils import parse_callback_params


def run_training(params, checkpoint_dir=None, recover=False, force=False):
    if checkpoint_dir is None:
        now = datetime.now()
        checkpoint_dir = f"train_logs/{now.strftime('%Y%m%d_%H%M%S')}"
    if force and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    logger.set_logger_dir(checkpoint_dir)

    # Load params
    data_params = params['data']
    dataflow_params = params['dataflow']
    models_params = params['models']
    callback_params = params['callbacks']
    trainer_params = params['trainer']
    params.to_file(os.path.join(checkpoint_dir, 'experiment.json'))

    # Fix seed for reproducable experiment
    fix_rng_seed(params['seed'])  # fix global seed in tensorpack
    tf.set_random_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    # Build dataflow
    data_dir = dataflow_params['data_dir']
    augmentor_params = dataflow_params['augmentors']
    if not bool(augmentor_params):
        augmentor_params = None

    if 'validation_split_ratio' in dataflow_params:
        validation_split_ratio = dataflow_params['validation_split_ratio']
    else:
        validation_split_ratio = None

    train_ds, val_ds = get_image_dataflow(
        data_dir, dataflow_params['batch_size'],
        dataflow_params['num_parallel'],
        augmentor_params=augmentor_params,
        validation_split_ratio=validation_split_ratio)

    # Build callbacks list
    extra_callbacks = parse_callback_params(callback_params, params)
    callbacks = [
        ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
        InferenceRunner(
            input=val_ds,
            infs=[ScalarStats('loss', prefix='val'),
                  ClassificationError(wrong_tensor_name='top1-error',
                                      summary_name='val-top1-error'),
                  ClassificationError(wrong_tensor_name='top3-error',
                                      summary_name='val-top3-error')]),
        MinSaver(monitor_stat='val-top3-error')
    ] + extra_callbacks

    # Build model
    models_params['image_size'] = data_params['image_size']
    models_params['num_labels'] = data_params['num_labels']
    models = BaseClassifier.from_params(models_params)

    # Build trainer
    trainer = SyncMultiGPUTrainerParameterServer(
        gpus=trainer_params["num_gpus"], ps_device=None)
    train_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=models,
        data=train_ds,
        callbacks=callbacks,
        steps_per_epoch=trainer_params["steps_per_epoch"],
        max_epoch=trainer_params["num_epochs"]
    )

    launch_train_with_config(train_config, trainer)
