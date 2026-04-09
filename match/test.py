import torch
from transformers import BertTokenizer
import time
from multiprocessing import Pool
import numpy as np


from .config import set_args
from .model import Model
from .utils import l2_normalize_2


args = set_args()
tokenizer = BertTokenizer.from_pretrained(args.pretrained_tokenizer_path)
# 加载训练好的神经网络
model = Model()
checkpoint = torch.load('match/outputs/base_model_epoch_4.bin', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()


def get_text_vec(text):
    inputs = tokenizer.encode_plus(
        text=text,
        text_pair=None,
        truncation=True,
        padding='max_length',
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=True
    )
    input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long)
    attention_mask = torch.tensor([inputs['attention_mask']], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, encoder_type=args.encoder_type)
    vec = l2_normalize_2(output.cpu().numpy())
    return vec


def get_match_score(jd_vec, cv):
    # TODO cv,jd记得到时候文本预处理一下preprocess
    cv_vec = get_text_vec(cv)
    sim = (cv_vec * jd_vec).sum()
    print("sim=", sim)
    return sim


def get_highest_match_score(jd_vec, *cvs: list):
    print("begin to get match score...")
    result_list = list()
    max_score = 0

    pool = Pool(3)
    for exp_list in cvs:
        for exp_txt in exp_list:
            result_list.append(pool.apply_async(get_match_score, (jd_vec, exp_txt,)))
    pool.close()
    pool.join()
    for res in result_list:
        score = res.get()
        max_score = max(max_score, score)
    return np.float64(max_score)






