import os
import re
import json

import tensorflow as tf


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load(file_name):
    with open(file_name, encoding='utf-8') as f:
        data = [line.rstrip('\n') for line in f]
        return data


def save(filename, data):
    with open(filename, encoding='utf-8', mode='w+') as f:
        for item in data:
            f.write(item + '\n')


def load_csv_with_index(file_name):
    data = {}
    with open(file_name) as f:
        for line in f:
            d = line.strip().split(',')
            data[int(d[0])] = d[1:]
    return data


def save_json(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            strs = json.dumps(item)
            f.write(str(strs) + '\n')


def append_json(f, data, close=False):
    """Append to an opened stream `f`"""
    assert not f.closed, "should only use this function when f is still opened"
    assert f.mode.startswith("a"), "should only use this function to append"
    for item in data:
        strs = json.dumps(item)
        f.write(str(strs) + '\n')
    if close and not f.closed:
        f.close()


def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)
