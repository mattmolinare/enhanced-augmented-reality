#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import sys
import ofp


def main(fname):
    params = ofp.read_yaml(fname)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        for fname in glob.iglob('*.yaml'):
            break
        else:
            raise IOError('no yaml file found')

    main(fname)
