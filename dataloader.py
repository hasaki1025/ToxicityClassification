
import pandas as pd
import numpy as np
import torch

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

padding_token = 'max_length'

class TextDataset(Dataset):
    def __init__(self, comments,labels, tokenizer, max_length):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        label = self.labels[idx]
        """编码后的返回一个tensor，shape = (1,max_length)"""
        encoding = self.tokenizer.encode_plus(comment,
                                        add_special_tokens=True,
                                        max_length=self.max_length,
                                        padding=padding_token,
                                        return_tensors='pt',
                                        truncation=True)
        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "label": torch.tensor(label),
        }


def create_dataloader(comments,labels,max_length,tokenizer,batch_size):
    dataset = TextDataset(comments,labels,tokenizer,max_length)
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)





def covert2bool(df,col_name):
    df[col_name] = df[col_name].fillna(0)
    df[col_name] = df[col_name] >= 0.5



def preprocess(df):
    #TODO 这样做的目的是什么
    for col in identity_columns + ['target']:
        covert2bool(df,col)
    comments = df['comment_text'].astype('str').tolist()
    labels = df['target'].tolist()
    return comments, labels


def load_jigsaw_data(logger,max_length,tokenizer,batch_size):
    train_data = pd.read_csv('data/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
    comments , label = preprocess(train_data)
    train_comments,valid_comments,train_label,valid_label= train_test_split(comments,label,test_size=0.2,random_state=42)
    #logger.info('{} train comments, {} validate comments' ,len(train_comments), len(valid_comments))
    print('%d train comments, %d validate comments' % (len(train_comments), len(valid_comments)))
    train_dataloader = create_dataloader(train_comments,train_label,max_length,tokenizer,batch_size)
    valid_dataloader = create_dataloader(valid_comments,valid_label,max_length,tokenizer,batch_size)
    return train_dataloader,valid_dataloader

