import os
import glob
import random
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List

class HTMLDataset(Dataset):
    def __init__(self, root_folder: str, tokenizer: any, mode: str, max_seq_len: int):
        super().__init__()

        self.root_folder = root_folder
        self.mode = mode
        self.max_seq_len = max_seq_len

        if mode != 'test':
            self.files = sorted(glob.glob(os.path.join(root_folder, f'*_{mode}.txt')))
            self.labels = None
            if mode == 'train':
                self.labels = [int(i[:-4].split('_')[-1]) for i in self.files]
        else:
            self.files = sorted(glob.glob(os.path.join(root_folder, '*_test.html')))
            self.labels = None

        self.tokenizer = tokenizer
        self._parse_xml()

    @property
    def _get_text(self, node: ET.Element) -> str:
        result = ''
        for child in node:
            if child.tag == '{http://www.w3.org/1999/xhtml}body':
                body = child
                break
        for el in body:
            result += ET.tostring(el, method='text', encoding='unicode').strip()
        return result

    def _parse_xml(self):
        """Parse XML files to retrieve URLs."""
        urls = {}
        for filename in self.files:
            tree = ET.parse(filename.replace('.txt', '.xml'))
            root = tree.getroot()
            urls[filename] = {'url': self._get_text(root), }

        self.urls = pd.DataFrame.from_dict(urls, orient='index', columns=['url'])

    def __len__(self):
        if self.labels is None:
            return len(self.files)
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == 'train' or self.mode == 'validation':
            txt_content = self.urls.loc[self.files[idx], 'url']
            true_label = self.labels[idx]

            sample = self.tokenizer.encode_plus(
                txt_content,
                max_length=self.max_seq_len,
                pad_to_max_length=True,
                truncation=True,
                return_tensors='pt',
            )

            return {
                'input_ids': sample['input_ids'][0],
                'attention_mask': sample['attention_mask'][0],
                'true_label': torch.tensor(true_label, dtype=torch.float16),
            }
        elif self.mode == 'test':
            txt_content = self.urls.loc[self.files[idx], 'url']

            sample = self.tokenizer.encode_plus(
                txt_content,
                max_length=self.max_seq_len,
                pad_to_max_length=True,
                truncation=True,
                return_tensors='pt',
            )

            return {
                'input_ids': sample['input_ids'][0],
                'attention_mask': sample['attention_mask'][0],
            }