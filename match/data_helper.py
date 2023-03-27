import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def split_dataset(y, ratio, seed=None):
    """划分训练集与测试集。这里只想得到相应的索引，因为想按照不同的形式分别加载训练集与测试集

    Parameters
    ----------
    y : pd.DataFrame
        label的那一列
    ratio : float
        训练集的比例，[0,1]之间的小数
    seed : int
        随机划分的种子

    Returns
    -------
        list, list
    """
    x = y
    x_train, x_test, _, _ = train_test_split(x, y, train_size=ratio,
                                             random_state=seed,
                                             shuffle=True,
                                             stratify=y)

    return x_train.index.tolist(), x_test.index.tolist()


def load_train_data(df, train_id):
    """加载训练集。按这种形式加载方便后续loss函数的编写

    Parameters
    ----------
    df : pd.DataFrame
        文本对，label
    ratio : list
        训练集的索引集合

    Returns
    -------
        list, list
    """
    sentence, label = [], []
    for id in train_id:
        sentence.extend([df.loc[id, 'cv'], df.loc[id, 'jd']])
        lab = int(df.loc[id, 'satisfied'])
        label.extend([lab, lab])

    return sentence, label


def load_test_data(df, test_id):
    sent1, sent2, label = [], [], []
    for id in test_id:
        sent1.append(df.loc[id, 'cv'])
        sent2.append(df.loc[id, 'jd'])
        label.append(df.loc[id, 'satisfied'])

    return sent1, sent2, label


# 重写DataLoader函数中的dataset形式
class CustomDataset(Dataset):
    def __init__(self, sentence, label, tokenizer):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index],
            text_pair=None,
            truncation=True,
            padding='max_length',
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }


# 重写DataLoader函数中的collate_fn函数。
# DataLoader过程：根据batch size得到一组采样索引；索引依次输入__getitem__()函数获得列表数据；列表数据传入collate_fn()得到最终该batch的输出形式。
def collate_fn(batch):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
        token_type_ids.append(item['token_type_ids'])
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)

    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids
