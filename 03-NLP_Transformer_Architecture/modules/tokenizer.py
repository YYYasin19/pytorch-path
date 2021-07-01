import pandas as pd
import numpy as np
import re

import unicodedata

class Tokenizer:
    
    def __init__(self):
        self.word_vector = {
            'sos': 0, 
            'eos': 1
        }
        self.n_words = 2
        
    def add_string(self, string:str):
        """adds this string to the word_vector"""
        
        tokens = self.tokenize_string(string)
        for token in tokens:
            self.word_vector[token] = self.n_words
            self.n_words += 1
            
    def add_data(self, data:pd.Series):
        strings = data.values.ravel()
        for s in strings:
            self.add_string(s)
            
    def tokenize_string(self, string):
        clean_s = self.clean_string(string)
        
        # just split 
        return clean_s.split(" ")
    
    
    @staticmethod
    def to_ascii(string):
        return ''.join(
            character for character in unicodedata.normalize('NFD', string) if unicodedata.category(character) != 'Mn'
        )
    
    @staticmethod
    def clean_string(string):
        s = Tokenizer.to_ascii(string.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s) # adds space before certain punctuation marks
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
        
    def __getitem__(self, index):
        # clean the index too
        c_index = Tokenizer.clean_string(index).strip()
        if c_index in self.word_vector:
            return self.word_vector[c_index]
        else:
            return None