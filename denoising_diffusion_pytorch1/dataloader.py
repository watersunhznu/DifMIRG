import torch
from torch.utils.data import Dataset
import os, re
import nltk
from denoising_diffusion_pytorch1.build_vocab import Vocabulary, build_vocab
import json
from PIL import Image
from torchvision import transforms
import jieba






# def create_captions(filepath, dataset):   #通过json文件生成caption
#     ## the captions have the impression and findings concatenated to form one big caption
#     ## i.e. caption = impression + " " + findings
#     ## WARNING: in addition to the XXXX in the captions, there are <unk> tokens
#
#     # clean for BioASQ
#     bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{},0-9]', '',
#                                 t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
#                                                                                               '').strip().lower()).split()
#
#     captions = []
#     '''
#     with open(filepath, "r") as file:
#
#         for line in file:
#             line = line.replace("\n", "").split("\t")
#
#             sentence_tokens = []
#
#             for sentence in line[1].split("."):
#                 tokens = bioclean(sentence)
#                 if len(tokens) == 0:
#                     continue
#                 caption = " ".join(tokens)
#                 sentence_tokens.append(caption)
#
#             captions.append(sentence_tokens)
#     '''
#     file = json.load(open(filepath, 'r',encoding='utf-8'))
#     if dataset == 'bra':
#         for item in file:
#             sentence_tokens = []
#
#             for sentence in item['paragraph'].split('。'):
#                 tokens = bioclean(sentence)
#                 if len(tokens) == 0:
#                     continue
#                 caption = " ".join(tokens)
#                 sentence_tokens.append(caption)
#
#             captions.append(sentence_tokens)
#     else:
#         for item in file:
#             sentence_tokens = []
#
#             for sentence in item['paragraph'].split('.'):
#                 tokens = bioclean(sentence)
#                 if len(tokens) == 0:
#                     continue
#                 caption = " ".join(tokens)
#                 sentence_tokens.append(caption)
#
#             captions.append(sentence_tokens)
#     return captions



import json
import re

def create_captions(filepath, dataset_type):
    """
    从指定的 JSON 文件中提取指定数据集（train/val/test）的 captions。
    - 针对 'bra' 数据集使用中文句号 "。" 分句；
    - 针对其他数据集使用英文句号 "." 分句；
    - 句子清理并拆分成 tokens 列表。

    Args:
        filepath (str): JSON 文件路径。
        dataset_type (str): 数据集类型，值应为 "train"、"val" 或 "test"。

    Returns:
        list: 包含 captions 的列表，每个元素为 tokenized sentences 列表。
    """
    def bioclean(text):
        """
        清理文本内容：
        - 移除标点符号、特殊字符、数字等；
        - 替换无效字符，转换为小写；
        - 移除多余的空格。
        """
        return re.sub(r'[.,?;*!%^&_+():\-\[\]{},0-9\u3002\uFF1F\uFF01\uFF0C\uFF1B\uFF1A]', '',
                      text.replace('"', '')
                          .replace('/', '')
                          .replace('\\', '')
                          .replace("'", '')
                          .strip()
                          .lower()).split()

    # 加载 JSON 文件
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file = json.load(f)
    except Exception as e:
        raise ValueError(f"加载 JSON 文件失败：{e}")

    # 检查 dataset_type 是否有效
    if dataset_type not in file:
        raise ValueError(f"数据集类型 '{dataset_type}' 无效。请确保其为 'train'、'val' 或 'test'。")

    # 选择分隔符：中文句号 "。" 或英文句号 "."
    sentence_splitter = "。" if dataset_type == 'bra' else "."

    # 提取指定数据集的段落
    captions = []
    for item in file[dataset_type]:
        # print('len(file[dataset_type])',len(file[dataset_type]))
        sentence_tokens = []

        # 分句并清理每个句子
        for sentence in item["paragraph"].split(sentence_splitter):
            tokens = bioclean(sentence)
            if len(tokens) == 0:  # 跳过空句子
                continue
            caption = " ".join(tokens)  # 拼接 tokens 为一个 caption
            sentence_tokens.append(caption)
            # print("sentence_tokens",sentence_tokens)

        captions.append(sentence_tokens)
    # print('captionscaptions',captions)

    return captions

class iuxray_train(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab=None, max_sent=4, max_word=32, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))

        tsv_file = os.path.join(self.root_dir, self.tsv_path)

        self.train_captions = create_captions("data1234.json", dataset_type="train")#读取语句，并分为一段一段的
        print('len(self.train_captions)',len(self.train_captions))
        #print("Train Captions:", self.train_captions)

        # self.captions = create_captions(tsv_file, 'iuxray')
        if vocab is None:
            self.vocab = build_vocab(self.train_captions, 1)
        else:
            self.vocab = vocab
        print('len(self.vocab)',len(self.vocab))
        # print('len(vocab)',len(vocab))
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        with open(tsv_file, 'r') as f:
            tsv_file = json.load(f)

        # 加载 train 数据
        self.data_file = tsv_file["train"]  # 直接获取 train 键对应的内容
        print('len(self.data_file)', len(self.data_file))
        self.max_sent = max_sent
        self.max_word = max_word
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['filename'])
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        caption = self.train_captions[idx]

        # print("---")
        # print(caption)#一句话分成多段的词语
        # print("----")

        sentences = []
        if self.max_sent >= len(caption) > 0:
            for i in range(self.max_sent):
                if i < len(caption):
                    tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                    # sentence = [self.vocab('<start>')]
                    sentence = []
                    sentence.extend([self.vocab(token) for token in tokens])
                    # sentence.append(self.vocab('<end>'))
                    # print([self.vocab.idx2word[k] for k in sentence])
                    sentences.append(sentence)
                else:
                    # sentence = [self.vocab('<start>'), self.vocab('<end>')]
                    sentence = []
                    sentences.append(sentence)
        elif self.max_sent < len(caption):#如果段落数超过max，直接舍弃后面的
            for i in range(self.max_sent):
                tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                # sentence = [self.vocab('<start>')]
                sentence = []
                sentence.extend([self.vocab(token) for token in tokens])
                # sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            # sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentence = []
            for i in range(self.max_sent):
                sentences.append(sentence)
            # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])

        # max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])

        for i in range(len(sentences)):
            if len(sentences[i]) < self.max_word:
                sentences[i] = sentences[i] + (self.max_word - len(sentences[i])) * [self.vocab('<pad>')]

        target = torch.Tensor(sentences)
        return image, target, len(caption)
class iuxray_test(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab=None, max_sent=4, max_word=32, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))

        tsv_file = os.path.join(self.root_dir, self.tsv_path)

        self.test_captions = create_captions("data1234.json", dataset_type="test")
        print('len(self.test_captions)',len(self.test_captions))
        # print("Test Captions:", self.test_captions)

        # self.captions = create_captions(tsv_file, 'iuxray')
        if vocab is None:
            self.vocab = build_vocab(self.test_captions, 1)
        else:
            self.vocab = vocab
        print('len(self.vocab)',len(self.vocab))
        # print('len(vocab)',len(vocab))
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        with open(tsv_file, 'r') as f:
            tsv_file = json.load(f)

        # 加载 test 数据
        self.data_file = tsv_file["test"]  # 直接获取 test 键对应的内容
        print('len(self.data_file)', len(self.data_file))
        self.max_sent = max_sent
        self.max_word = max_word
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['filename'])
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        caption = self.test_captions[idx]

        # print("---")
        # print(caption)
        # print("----")

        sentences = []
        if self.max_sent >= len(caption) > 0:
            for i in range(self.max_sent):
                if i < len(caption):
                    tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                    # sentence = [self.vocab('<start>')]
                    sentence = []
                    sentence.extend([self.vocab(token) for token in tokens])
                    # sentence.append(self.vocab('<end>'))
                    # print([self.vocab.idx2word[k] for k in sentence])
                    sentences.append(sentence)
                else:
                    # sentence = [self.vocab('<start>'), self.vocab('<end>')]
                    sentence = []
                    sentences.append(sentence)
        elif self.max_sent < len(caption):
            for i in range(self.max_sent):
                tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                # sentence = [self.vocab('<start>')]
                sentence = []
                sentence.extend([self.vocab(token) for token in tokens])
                # sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            # sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentence = []
            for i in range(self.max_sent):
                sentences.append(sentence)
            # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])

        # max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])

        for i in range(len(sentences)):
            if len(sentences[i]) < self.max_word:
                sentences[i] = sentences[i] + (self.max_word - len(sentences[i])) * [self.vocab('<pad>')]

        target = torch.Tensor(sentences)
        return image, target, len(caption)


class iuxray_val(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab=None, max_sent=4, max_word=32, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))

        tsv_file = os.path.join(self.root_dir, self.tsv_path)

        self.val_captions = create_captions("data1234.json", dataset_type="val")
        print('len(self.val_captions)',len(self.val_captions))
        # print("Val Captions:", self.val_captions)

        # self.captions = create_captions(tsv_file, 'iuxray')
        if vocab is None:
            self.vocab = build_vocab(self.val_captions, 1)
        else:
            self.vocab = vocab
        print('len(self.vocab)',len(self.vocab))
        # print('len(vocab)',len(vocab))
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        with open(tsv_file, 'r') as f:
            tsv_file = json.load(f)

        # 加载 val 数据
        self.data_file = tsv_file["val"]  # 直接获取 val 键对应的内容
        print('len(self.data_file)', len(self.data_file))
        self.max_sent = max_sent
        self.max_word = max_word
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['filename'])
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        caption = self.val_captions[idx]

        # print("---")
        # print(caption)
        # print("----")

        sentences = []
        if self.max_sent >= len(caption) > 0:
            for i in range(self.max_sent):
                if i < len(caption):
                    tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                    # sentence = [self.vocab('<start>')]
                    sentence = []
                    sentence.extend([self.vocab(token) for token in tokens])
                    # sentence.append(self.vocab('<end>'))
                    # print([self.vocab.idx2word[k] for k in sentence])
                    sentences.append(sentence)
                else:
                    # sentence = [self.vocab('<start>'), self.vocab('<end>')]
                    sentence = []
                    sentences.append(sentence)
        elif self.max_sent < len(caption):
            for i in range(self.max_sent):
                tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                # sentence = [self.vocab('<start>')]
                sentence = []
                sentence.extend([self.vocab(token) for token in tokens])
                # sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            # sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentence = []
            for i in range(self.max_sent):
                sentences.append(sentence)
            # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])

        # max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])

        for i in range(len(sentences)):
            if len(sentences[i]) < self.max_word:
                sentences[i] = sentences[i] + (self.max_word - len(sentences[i])) * [self.vocab('<pad>')]

        target = torch.Tensor(sentences)
        return image, target, len(caption)


class mimic(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab=None, max_sent=10, max_word=50, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))

        tsv_file = os.path.join(self.root_dir, self.tsv_path)

        self.captions = create_captions(tsv_file, 'mimic')
        if vocab is None:
            self.vocab = build_vocab(self.captions, 10)
        else:
            self.vocab = vocab
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.data_file = json.load(open(tsv_file, 'r'))
        self.transform = transform
        # self.tags_l = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
        #                'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
        #                'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

        # self.tags_l = ['normal', 'opacity', 'degenerative change', 'atelectases', 'atelectasis', 'scarring',
        #                'cardiomegaly', 'calcified granuloma', 'granuloma', 'pneumonia',  'others']
        self.max_sent = max_sent
        self.max_word = max_word
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        # print(idx)
        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['file_name'])
        image = Image.open(img_name)
        #
        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions[idx]

        # print("---")
        # print(caption)
        # print("----")

        sentences = []
        if self.max_sent >= len(caption) > 0:
            for i in range(self.max_sent):
                if i < len(caption):
                    tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                    # sentence = [self.vocab('<start>')]
                    sentence = []
                    sentence.extend([self.vocab(token) for token in tokens])
                    # sentence.append(self.vocab('<end>'))
                    # print([self.vocab.idx2word[k] for k in sentence])
                    sentences.append(sentence)
                else:
                    # sentence = [self.vocab('<start>'), self.vocab('<end>')]
                    sentence = []
                    sentences.append(sentence)
        elif self.max_sent < len(caption):
            for i in range(self.max_sent):
                tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())[:self.max_word - 2]
                # sentence = [self.vocab('<start>')]
                sentence = []
                sentence.extend([self.vocab(token) for token in tokens])
                # sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            # sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentence = []
            for i in range(self.max_sent):
                sentences.append(sentence)
            # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])

        # max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])

        for i in range(len(sentences)):
            if len(sentences[i]) < self.max_word:
                sentences[i] = sentences[i] + (self.max_word - len(sentences[i])) * [self.vocab('<pad>')]

        target = torch.Tensor(sentences)
        return image, target, len(caption)


class bra(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab=None, transform=None, max_sent=4, max_word=32):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))

        tsv_file = os.path.join(self.root_dir, self.tsv_path)

        self.captions = create_captions(tsv_file, 'bra')
        if vocab is None:
            self.vocab = build_vocab(self.captions, 1, 'ch')
        else:
            self.vocab = vocab
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.data_file = json.load(open(tsv_file, 'r'))
        self.transform = transform
        self.tags_l = ['淋巴结增多', '混合型', '片絮状影', '结节影', '腺体影', '腺体致密', '致密灶', '轻度增大', '钙化灶', '高密度影']
        jieba.add_word('高密度影')
        jieba.add_word('片絮状影')
        self.max_sent = max_sent
        self.max_word = max_word
        # self.tags_l = ['normal', 'opacity', 'degenerative change', 'atelectases', 'atelectasis', 'scarring',
        #                'cardiomegaly', 'calcified granuloma', 'granuloma', 'pneumonia',  'others']

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        # print(idx)
        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['file_name'])
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions[idx]

        # print("---")
        # print(caption)
        # print("----")
        sentences = []
        if self.max_sent >= len(caption) > 0:
            for i in range(self.max_sent):
                if i < len(caption):
                    tokens = jieba.cut(caption[i])
                    # sentence = [self.vocab('<start>')]
                    sentence = []
                    sentence.extend([self.vocab(token) for token in tokens][:self.max_word - 2])
                    # sentence.append(self.vocab('<end>'))
                    # print([self.vocab.idx2word[k] for k in sentence])
                    sentences.append(sentence)
                else:
                    # sentence = [self.vocab('<start>'), self.vocab('<end>')]
                    sentence = []
                    sentences.append(sentence)
        elif self.max_sent < len(caption):
            for i in range(self.max_sent):
                tokens = jieba.cut(caption[i])
                # sentence = [self.vocab('<start>')]
                sentence = []
                sentence.extend([self.vocab(token) for token in tokens][:self.max_word - 2])
                # sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            # sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentence = []
            for i in range(self.max_sent):
                sentences.append(sentence)
            # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])

        # max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])

        for i in range(len(sentences)):
            if len(sentences[i]) < self.max_word:
                sentences[i] = sentences[i] + (self.max_word - len(sentences[i])) * [self.vocab('<pad>')]

        target = torch.Tensor(sentences)

        return image, target, len(sentences)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, no_of_sent, max_sent_len).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption, no_of_sent, max_sena list indicating valid length for each caption.t_len).
            - image: torch tensor of shape (3, crop_size, crop_size).
            - caption: torch tensor of shape (no_of_sent, max_sent_len); variable length.
            - no_of_sent: number of sentences in the caption
            - max_sent_len: maximum length of a sentence in the caption

    Returns:
        images: torch tensor of shape (batch_size, 3, crop_size, crop_size).
        targets: torch tensor of shape (batch_size, max_no_of_sent, padded_max_sent_len).
        prob: torch tensor of shape (batch_size, max_no_of_sent)
    """
    # Sort a data list by caption length (descending order).
    #     data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, len_sentences = zip(*data)
    max_sent, max_word = captions[0].shape

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    # tags = torch.LongTensor(tags_yn)
    targets = torch.zeros(len(captions), max_sent, max_word).long()
    prob = torch.zeros(len(captions), max_sent).long()

    for i, cap in enumerate(captions):
        for j, sent in enumerate(cap):
            try:
                targets[i, j, :len(sent)] = sent[:]
            except:
                print('d')
            else:
                if j < len_sentences[i]:
                    prob[i, j] = 1
        # stop after the last sentence
        # prob[i, j] = 0

    return images, targets, prob


def get_loader(root_dir, tsv_path, image_path, transform, batch_size, shuffle, num_workers, dataset, max_sent,
               max_word, vocab=None, drop_last=False, mode='train'):  # 新增 drop_last 参数
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # dataset
    if mode == 'train':
        if dataset == 'iuxray':
            data = iuxray_train(root_dir=root_dir,
                                tsv_path=tsv_path,
                                image_path=image_path,
                                vocab=vocab,
                                transform=transform,
                                max_sent=max_sent,
                                max_word=max_word
                                )
        elif dataset == 'mimic':
            data = mimic(root_dir=root_dir,
                         tsv_path=tsv_path,
                         image_path=image_path,
                         vocab=vocab,
                         transform=transform,
                         max_sent=max_sent,
                         max_word=max_word
                         )
        elif dataset == 'bra':
            data = bra(root_dir=root_dir,
                       tsv_path=tsv_path,
                       image_path=image_path,
                       vocab=vocab,
                       transform=transform,
                       max_sent=max_sent,
                       max_word=max_word
                       )
    elif mode == 'test':
        if dataset == 'iuxray':
            data = iuxray_test(root_dir=root_dir,
                                tsv_path=tsv_path,
                                image_path=image_path,
                                vocab=vocab,
                                transform=transform,
                                max_sent=max_sent,
                                max_word=max_word
                                )
        elif dataset == 'mimic':
            data = mimic(root_dir=root_dir,
                         tsv_path=tsv_path,
                         image_path=image_path,
                         vocab=vocab,
                         transform=transform,
                         max_sent=max_sent,
                         max_word=max_word
                         )
        elif dataset == 'bra':
            data = bra(root_dir=root_dir,
                       tsv_path=tsv_path,
                       image_path=image_path,
                       vocab=vocab,
                       transform=transform,
                       max_sent=max_sent,
                       max_word=max_word
                       )
    elif mode == 'val':
        if dataset == 'iuxray':
            data = iuxray_val(root_dir=root_dir,
                                tsv_path=tsv_path,
                                image_path=image_path,
                                vocab=vocab,
                                transform=transform,
                                max_sent=max_sent,
                                max_word=max_word
                                )
        elif dataset == 'mimic':
            data = mimic(root_dir=root_dir,
                         tsv_path=tsv_path,
                         image_path=image_path,
                         vocab=vocab,
                         transform=transform,
                         max_sent=max_sent,
                         max_word=max_word
                         )
        elif dataset == 'bra':
            data = bra(root_dir=root_dir,
                       tsv_path=tsv_path,
                       image_path=image_path,
                       vocab=vocab,
                       transform=transform,
                       max_sent=max_sent,
                       max_word=max_word
                       )
    print(f"Dataset size: {len(data)}")  # 输出数据集大小
    assert len(data) > 0, "Dataset is empty. Please check your data."

    # Data loader for dataset
    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              drop_last=drop_last,  # 添加 drop_last 参数
                                              collate_fn=collate_fn)


    return data_loader, data.vocab



if __name__ == '__main__':
    root_dir = '/home/mzjs/data'
    tsv_path = 'data.json'
    image_path = 'images'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    train_loader, _ = get_loader(root_dir, tsv_path, image_path, transform, 16, True, 1, 'bra', max_sent=4,
                                 max_word=32)
    for i in range(10):
        print(_.idx2word[i])
    for i, (img, captions, prob) in enumerate(train_loader):
        print(captions)
