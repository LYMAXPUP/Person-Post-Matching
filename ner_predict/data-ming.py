#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2020-01-09 18:51
"""
import re
import json


def prepare_label():
    text = """
    教育经历（education）: 544
    项目经验（experience）: 290
    软技能（soft skills）: 258
    专业技能（professional skills）: 479
    """

    a = re.findall(r"（(.*?)）", text.strip())
    print(a)
    label2id = {"O": 0}
    index = 1
    for i in a:
        label2id["S_" + i] = index
        label2id["B_" + i] = index + 1
        label2id["M_" + i] = index + 2
        label2id["E_" + i] = index + 3
        index += 4

    open("label2id.json", "w").write(json.dumps(label2id, ensure_ascii=False, indent=2))


def prepare_len_count():
    len_count = {}

    for line in open("data/thuctc_train.json"):
        if line.strip():
            _ = json.loads(line.strip())
            len_ = len(_["text"])
            if len_count.get(len_):
                len_count[len_] += 1
            else:
                len_count[len_] = 1

    for line in open("data/thuctc_valid.json"):
        if line.strip():
            _ = json.loads(line.strip())
            len_ = len(_["text"])
            if len_count.get(len_):
                len_count[len_] += 1
            else:
                len_count[len_] = 1

    print("len_count", json.dumps(len_count, indent=2))
    open("len_count.json", "w").write(json.dumps(len_count, indent=2))


def label_count(path):
    labels = ['education', 'experience','publications', 'softskills', 'professionalskills']
    label2desc = {
        "education": "教育经历",
        "experience": "项目经验",
        "publications": "证书资质",
        "soft skills": "软技能",
        "professional skills": "专业能力",
    }
    label_count_dict = {i: 0 for i in labels}
    for line in open(path):
        if line.strip():
            _ = json.loads(line.strip())
            for k, v in _["label"].items():
                label_count_dict[k] += len(v)
    for k, v in label_count_dict.items():
        print("{}（{}）:{}".format(label2desc[k], k, v))
    print("\n")


# prepare_label()
# label_count("data/thuctc_train.json")
# label_count("data/thuctc_valid.json")
