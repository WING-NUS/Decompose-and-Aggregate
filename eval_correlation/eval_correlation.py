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


import sys
sys.path.append("../")

import os
import argparse
import numpy as np
import kendall
from LLMEval.data_utils import dict2json, json2dict, readcsv, str2list, strclean


DATA_DIR = "./data/Weighting_Eval/"
OUTPUT_DIR = "./data/output/"
datasets = ["faireval", "mtbench", "llmbar", "instrusum"]
models = ["gpt4", "gpt35", "llama2", "mistral"]


def data_preprocess(args, task_name):
    data_root = args.data_dir + task_name + "/"
    dict_temp = {"gpt4":[], "gpt35":[], "llama2":[], "mistral":[]}
    dict_temp_1 = {"gpt4": [], "gpt35": [], "llama2": [], "mistral": [], "mistralv01": []}
    dict_ranks = {task_name: dict_temp}
    if task_name == "instrusum":
        dict_ranks = {task_name: dict_temp_1}
    for root, _dirnames, filenames in os.walk(data_root):
        # print(data_root, filenames)
        for filename in filenames:
            if filename.rfind(".csv") != len(filename) - 4:
                continue
            else:
                for llm in models:
                    if llm in filename:
                        current_llm = llm
                    if "mistralv01" in filename:
                        current_llm = "mistralv01"
            raw = readcsv(data_root + filename)
            ranks = []
            if task_name == "faireval":
                """
                text,aspect1,aspect2,aspect3,aspect4,accuracy_w_gpt4,helpfulness_w_gpt4,relevance_w_gpt4,detail_w_gpt4,ann1,ann2
                LLM results: col 5 - 8
                human results: col 9, 10 (we use 9) 
                """
                for line in raw:
                    # print(line)
                    list_llm = [
                        float(line["accuracy_w_" + current_llm]),
                        float(line["helpfulness_w_" + current_llm]),
                        float(line["relevance_w_" + current_llm]),
                        float(line["detail_w_" + current_llm]),
                    ]
                    list_human = str2list(line["ann1"], ",")
                    list_human_2 = str2list(line["ann2"], ",")
                    rank_llm = rank_scores(list_llm)
                    rank_human = rank_scores(list_human)
                    rank_human_2 = rank_scores(list_human_2)
                    ranks.append([rank_llm, rank_human, rank_human_2])
            elif (
                task_name == "llmbar"
            ):
                """
                text,aspect1,aspect2,aspect3,ann1,ann2,gpt4_w1,gpt4_w2,gpt4_w3
                LLM results: col 6 - 8
                human results: col 4, 5 (we use 4) 
                """
                for line in raw:
                    if line[current_llm + "_w1"] == "":
                        continue 
                    list_llm = [
                        float(strclean(line[current_llm + "_w1"])),
                        float(strclean(line[current_llm + "_w2"])),
                        float(strclean(line[current_llm + "_w3"])),
                    ]
                    list_human = str2list(line["ann1"], ",")
                    list_human_2 = str2list(line["ann2"], ",")
                    rank_llm = rank_scores(list_llm)
                    rank_human = rank_scores(list_human)
                    rank_human_2 = rank_scores(list_human_2)
                    ranks.append([rank_llm, rank_human, rank_human_2])
            elif task_name == "mtbench":
                """
                text,aspect1,aspect2,aspect3,aspect4,aspect5,aspect6,creativity_w,helpfulness_w,accuracy_w,depth_w,detail_w,relevance_w,ann1,ann2
                LLM results: col 7 - 12
                human results: col 13, 14 (we use 13) 
                """  
                for line in raw:
                    # print(line)
                    if current_llm == "gpt4": 
                        list_llm = [
                            float(line["creativity_w"]),
                            float(line["helpfulness_w"]),
                            float(line["accuracy_w"]),
                            float(line["depth_w"]),
                            float(line["detail_w"]),
                            float(line["relevance_w"]),
                        ]
                    else:
                        list_llm = [
                            float(line["creativity_w_" + current_llm]),
                            float(line["helpfulness_w_" + current_llm]),
                            float(line["accuracy_w_" + current_llm]),
                            float(line["depth_w_" + current_llm]),
                            float(line["detail_w_" + current_llm]),
                            float(line["relevance_w_" + current_llm]),
                        ]
                    list_human = str2list(line["ann1"], ",")
                    list_human_2 = str2list(line["ann2"], ",")
                    rank_llm = rank_scores(list_llm)
                    rank_human = rank_scores(list_human)
                    rank_human_2 = rank_scores(list_human_2)
                    ranks.append([rank_llm, rank_human, rank_human_2])
            elif task_name == "instrusum":
                """
                text,gpt4_metrics,aspect1,aspect2,aspect3,gpt4_w1,gpt4_w2,gpt4_w3,ann1,ann2
                LLM results: col 5 - 7
                human results: col 8, 9 (we use 8) 
                article,annotations,requirement,text,aspects,aspect1,aspect2,aspect3,gpt35_w1,gpt35_w2,gpt35_w3,ann1,ann2
                LLM results: col 8 - 10
                human results: col 11, 12 (we use 11)
                text,aspect1,aspect2,aspect3,llama2_w_r,ann1,ann2
                text,aspect1,aspect2,aspect3,mistral_w_r,ann1,ann2 
                LLM results: col 4
                human results: col 5, 6 (we use 5) 
                """
                for line in raw:
                    # print(line)
                    if current_llm in ["gpt4", "gpt35"]:
                        list_llm = [
                            float(strclean(line[current_llm + "_w1"])),
                            float(strclean(line[current_llm + "_w2"])),
                            float(strclean(line[current_llm + "_w3"])),
                        ]
                    elif current_llm in ["llama2", "mistral"]:
                        list_llm = str2list(line[current_llm + "_w_r"], " ")
                    else: # current_llm == "mistralv01"
                        list_llm = str2list(line["mistral" + "_w_r"], " ")
                    list_human = str2list(line["ann1"], ",")
                    list_human_2 = str2list(line["ann2"], ",")
                    rank_llm = rank_scores(list_llm)
                    rank_human = rank_scores(list_human)
                    rank_human_2 = rank_scores(list_human_2)
                    ranks.append([rank_llm, rank_human, rank_human_2])
            dict_ranks[task_name][current_llm] = ranks
    return dict_ranks 


def rank_scores(lis):
    # lis = np.array([1,2,1,8,7])
    sorted_index = np.argsort(lis)
    # print(sorted_index)
    for x in lis: 
        inx = [i for i,val in enumerate(lis) if val==x]
        for i in range(len(inx)-1):
            sorted_index[i] = sorted_index[inx[-1]] 
    # print(sorted_index)
    rank = [len(sorted_index) - a for a in sorted_index]
    # print(rank)
    return rank 


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=DATA_DIR,
        type=str,
        # required=True,
        help="The input data directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(datasets) + ". Or all",
    )

    args = parser.parse_args()

    dict_ranks = dict()
    if args.task_name != "all":
        dict_ranks = data_preprocess(args, args.task_name)
    else:
        for task_name in datasets:
            print(task_name)
            dict_ranks.update(data_preprocess(args, task_name))
    # print(dict_ranks)
    dict2json(dict_ranks, args.output_dir + "ranks.json")

    # dict_ranks = json2dict(args.output_dir + "ranks.json")

    tasks = datasets if args.task_name == "all" else [args.task_name]
    dict_cors = {
        "faireval": {},
        "mtbench": {},
        "llmbar": {},
        "instrusum": {},
    }
    for task in tasks:
        # print(task)
        if task == "instrusum":
            models = ["gpt4", "gpt35", "llama2", "mistral", "mistralv01"]
        else:
            models = ["gpt4", "gpt35", "llama2", "mistral"]
        list_cor_2humans = []
        for llm in models:
            # print(llm)
            list_ranks = dict_ranks[task][llm]
            list_cor_llmhuman = []
            list_cor_2humans_for_llm = []
            for rank_pair in list_ranks:
                list_cor_llmhuman.append(
                    kendall.kendall_top_k(
                        np.array(rank_pair[0]), np.array(rank_pair[1])
                    )
                )
                list_cor_2humans.append(
                    kendall.kendall_top_k(
                        np.array(rank_pair[1]), np.array(rank_pair[2])
                    )
                )
                list_cor_2humans_for_llm.append(
                    kendall.kendall_top_k(
                        np.array(rank_pair[1]), np.array(rank_pair[2])
                    )
                )
            dict_cors[task][llm + "-human"] = np.mean(list_cor_llmhuman)
            print(task, "\t", llm, "- human\t", np.mean(list_cor_llmhuman))
            print(
                task,
                "\t",
                llm,
                "\t",
                "human - human\t",
                np.mean(list_cor_2humans_for_llm),
            )
        dict_cors[task]["human-human"] = np.mean(list_cor_2humans)
        print(task, "\t", "human - human\t", np.mean(list_cor_2humans))
    print(dict_cors)
    dict2json(dict_cors, args.output_dir + "correlations.json")

if __name__ == "__main__":
    main()
