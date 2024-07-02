# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024, Shumin Deng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import codecs
import csv
import logging
import os
from typing import List

import tqdm


logger = logging.getLogger(__name__)


def json2dicts(jsonFile):
    data = []
    with codecs.open(jsonFile, "r", "utf-8") as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    return data


def json2dict(jsonFile):
    data = []
    with codecs.open(jsonFile, "r", "utf-8") as f:
        for line in f:
            dic = json.loads(line)
    return dic


def dict2json(dic, jsonFile):
    with open(jsonFile, 'w') as outfile: # 'a+'
        json.dump(dic, outfile, default=int)
        # outfile.write('\n')
        print("Finishing writing a dict into " + jsonFile)


def readcsv(csvfile):
    data = []
    with codecs.open(csvfile, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            # print(row)
            data.append(row)
    f.close()
    return data

def str2list(str_lis, split_str):
    str_lis = strclean(str_lis)
    list_temp = (
        str_lis.replace("[", "").replace("]", "").replace("'", "").split(split_str)
    )
    lis = [float(a.strip()) for a in list_temp]
    return lis


def strclean(str_original):
    str_clean = str_original.replace("%", "").replace("$", "").replace("'", "")
    return str_clean 
