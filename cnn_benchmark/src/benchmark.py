import os
import time
import shutil
import subprocess
import tensorflow as tf

from datetime import datetime


def get_iter_value_array(nrof_values):
    num_values = len(nrof_values)
    iter_array = []
    iter_value = [-1] * num_values
    idx = 0
    while True:
        iter_value[idx] += 1
        if iter_value[idx] == nrof_values[idx]:
            if idx == 0:
                break
            else:
                iter_value[idx] = -1
                idx -= 1
        elif idx == num_values - 1:
            iter_array.append(iter_value.copy())
        else:
            idx += 1
    return iter_array


def parse_benchmark_params(iter_params, params):
    for path, values in iter_params.items():
        levels = path.split('.')
        dict_ref = [params[levels[0]]]
        for i in range(1, len(levels) - 1):
            dict_ref.append(dict_ref[i - 1][levels[i]])
        dict_ref[-1][levels[-1]] = values
    return params


def run_benchmark(params, checkpoint_dir, show_logs=False):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load params
    benchmark_params = params['benchmark']
    nrof_values = []
    for name, list_value in benchmark_params.items():
        nrof_values.append(len(list_value))
    iter_value_array = get_iter_value_array(nrof_values)

    for i, iter_values in enumerate(iter_value_array):
        iter_checkpoint_dir = os.path.join(
            checkpoint_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        iter_params = {}
        idx = 0
        for param, param_list in benchmark_params.items():
            iter_params[param] = param_list[iter_values[idx]]
            idx += 1
        params = parse_benchmark_params(iter_params, params)
        iter_config_file = os.path.abspath(os.path.join(
            checkpoint_dir, '.tmp.json'))
        params.to_file(iter_config_file)

        print(f'Start iter {i + 1}: {iter_params}')
        start_time = time.time()
        p = subprocess.Popen(
            f'python3 run.py train {iter_config_file} '
            f'-c {os.path.abspath(iter_checkpoint_dir)}',
            stdout=subprocess.PIPE if show_logs else subprocess.DEVNULL,
            stderr=subprocess.PIPE if show_logs else subprocess.DEVNULL,
            shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        print(f'Status: {p_status} - Time: {time.time() - start_time}')
