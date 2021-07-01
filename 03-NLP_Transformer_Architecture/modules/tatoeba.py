import pandas as pd
import numpy as np
import re

import torch
from torch.utils.data import Dataset
from .tokenizer import Tokenizer

class TatoebaDataset(Dataset):
    
    def __init__(self, df:pd.DataFrame, max_length=None, langs=('en', 'de'), debug=False):
        self.debug = debug
        self.df = df.copy(deep=True)
        self.langs = langs
        
        # filter by max length
        self.df = self.df.loc[df[langs[0]].str.len() <= max_length, :]
        
        # clean all strings -> add 'clean'-versions of language data!
        self.df.loc[:, langs[0] + '_clean'] = self.df.loc[:, langs[0]].apply(Tokenizer.clean_string)
        self.df.loc[:, langs[1] + '_clean'] = self.df[langs[1]].apply(Tokenizer.clean_string)
        
        # create the tokenizers for both languages
        self.t0 = Tokenizer()
        self.t0.add_data(self.df[langs[0] + '_clean'])
        self.t1 = Tokenizer()
        self.t1.add_data(self.df[langs[1] + '_clean'])
    
    def __getitem__(self, index):
        if index > self.__len__():
            return None
        
        # get sentences in both languages from line <index>
        s0, s1 = self.df.loc[index, [self.langs[0] + '_clean', self.langs[1] + '_clean']]
        
        # vectorize sentences
        if self.debug: print(f" >> {self.langs[0]}: {s0}, {self.langs[1]} {s1}")
        if self.debug: print(f" >> Tokenized: {s0.split(' ')}")
        
        v0 = [self.t0[s] for s in s0.split(" ") if s != ""]
        v1 = [self.t1[s] for s in s1.split(" ") if s != ""]
        
        # transform to tensors, .view(-1,1) just ensures that we have 1-d vectors
        t0 = torch.tensor(v0).view(-1,1)
        t1 = torch.tensor(v1).view(-1,1)
        
        return t0, t1
    
    def __len__(self):
        return self.df.shape[0]
        pass
    pass