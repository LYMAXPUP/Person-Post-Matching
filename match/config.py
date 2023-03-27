import argparse


def set_args():
    parser = argparse.ArgumentParser('--CoSENT进行相似性判断')
    parser.add_argument('--pretrained_tokenizer_path', default='match/RoBERTa_pretrain', type=str, help='预训练分词')
    parser.add_argument('--pretrained_config_path', default='match/RoBERTa_pretrain/config.json', type=str, help='预训练配置文件')
    parser.add_argument('--pretrained_model_path', default='match/RoBERTa_pretrain/pytorch_model.bin', type=str, help='预训练模型')

    # parser.add_argument('--pretrained_tokenizer_path', default='./mengzi_pretrain', type=str, help='预训练分词')
    # parser.add_argument('--pretrained_config_path', default='./mengzi_pretrain/config.json', type=str, help='预训练配置文件')
    # parser.add_argument('--pretrained_model_path', default='./mengzi_pretrain/pytorch_model.bin', type=str, help='预训练模型')

    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='训练几轮')
    parser.add_argument('--train_batch_size', default=8, type=int, help='训练批次大小')
    parser.add_argument('--val_batch_size', default=8, type=int, help='测试批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--encoder_type', default='fist-last-avg', type=str, help='输出层的编码方式')
    parser.add_argument('--seed', default=43, type=int, help='随机种子')

    return parser.parse_args()
