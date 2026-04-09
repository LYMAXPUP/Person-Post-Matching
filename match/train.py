import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.models.bert import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from preprocess import *
from .config import set_args
from .model import Model
from .utils import l2_normalize, compute_corrcoef
from .data_helper import CustomDataset, collate_fn, load_train_data, load_test_data, split_dataset


args = set_args()
tokenizer = BertTokenizer.from_pretrained(args.pretrained_tokenizer_path)


# 设置模型随机初始化的种子
def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


# 句子先经过tokenizer编码，后续输入model
def get_sent_id_tensor(s_list):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    for s in s_list:
        inputs = tokenizer.encode_plus(
            text=s,
            text_pair=None,
            truncation=True,
            padding='max_length',
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])
        token_type_ids.append(inputs['token_type_ids'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

    return all_input_ids, all_attention_mask, all_token_type_ids


# 计算测试集的准确率
def evaluate(df, test_id):
    sent1, sent2, label = load_test_data(df, test_id)
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    model.eval()
    for s1, s2, lab in tqdm(zip(sent1, sent2, label)):
        input_ids, input_mask, segment_ids = get_sent_id_tensor([s1, s2])
        lab = torch.tensor([lab], dtype=torch.float)
        if torch.cuda.is_available():
            input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
            lab = lab.cuda()

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type=args.encoder_type)

        all_a_vecs.append(output[0].cpu().numpy())
        all_b_vecs.append(output[1].cpu().numpy())
        all_labels.extend(lab.cpu().numpy())

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)

    return corrcoef


def calc_loss(y_true, y_pred):
    # 1. 取出真实的标签
    y_true = y_true[::2]    # tensor([1, 0, 1]) 真实的标签

    # 2. 对输出的句子向量进行l2归一化，这样两个向量相乘直接就是计算余弦相似度
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_pred = y_pred / norms
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20

    # 3. 取出负例与正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置两两之间余弦的差值
    # 矩阵中的第i行j列，表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]   # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        
    return torch.logsumexp(y_pred, dim=0)


if __name__ == '__main__':
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)

    # 数据预处理
    df = pd.read_csv('./data/paws.csv')
    # df = pd.read_csv('./data/dataset.csv')
    # for i in df.index:
    #     df.loc[i, 'cv'] = remove_tokens(df.loc[i,'cv'])
    #     df.loc[i, 'jd'] = filter_points(remove_tokens(df.loc[i, 'cv']), key_path='./data/key.txt')

    # 划分数据集
    train_id, test_id = split_dataset(df['satisfied'], 0.9)   # 9:1
    train_sentence, train_label = load_train_data(df, train_id)
    train_dataset = CustomDataset(sentence=train_sentence, label=train_label, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=1)

    total_steps = len(train_dataloader) * args.num_train_epochs

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    for epoch in range(args.num_train_epochs):
        model.train()
        train_label, train_predict = [], []
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            if torch.cuda.is_available():
                input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                label_ids = label_ids.cuda()
            output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type='fist-last-avg')
            loss = calc_loss(label_ids, output)
            loss.backward()
            print("当前轮次:{}, 正在迭代:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
            epoch_loss += loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        corr = evaluate(df, test_id)
        s = 'Epoch:{} | corr: {:10f}'.format(epoch, corr)
        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            s += '\n'
            f.write(s)

        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.output_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)   # 存model的参数
