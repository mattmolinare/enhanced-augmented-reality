# -*- coding: utf-8 -*-

import yaml

__all__ = ['read_yaml']


def read_yaml(fname):
    with open(fname, 'r') as fp:
        data = yaml.load(fp)
    return data
