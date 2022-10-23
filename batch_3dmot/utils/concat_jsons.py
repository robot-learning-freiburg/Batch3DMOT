#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2022
@author: Martin Buechner, buechner@cs.uni-freiburg.de
"""
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load config.yaml to batch_3dmot project',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file1', type=str, help='Provide first JSON.')
    parser.add_argument('--file2', type=str, help='Provide second JSON.')
    parser.add_argument('--outfile', type=str, help='Provide output JSON filename')
    opt = parser.parse_args()

    with open(opt.file1, 'r') as f1:
        json_one = json.load(f1)

    with open(opt.file2, 'r') as f2:
        json_two = json.load(f2)

    print(json_one.keys())
    print(json_two.keys())

    """"""
    for split, _ in json_one.items():
        json_one[split].extend(json_two[split]) # extend or update
    
    #json_one.update(json_two)

    with open(opt.outfile, 'w') as outfile:
        json.dump(json_one, outfile)