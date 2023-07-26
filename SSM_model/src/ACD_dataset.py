import torch
import pandas as pd
import os

from utils import jsonlload
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}
tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-small-korean-summarization')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
def tokenize_and_align_labels(form, label,style):
    
    global polarity_count
    global entity_property_count
    ssm_encode_data_dict = {
        'input_ids': [],
        'attention_mask': []
    }
    ssm_decode_data_dict = {
        'input_ids': [],
        'attention_mask': []
    }
    #bart모델의 경우 BOS EOS토큰 넣어줘야함
    #T5의경우 <pad>토큰 넣어줘야함
    answer_label = "<pad>"+ style + ',' + label
    sentence = 'summary: ' + form

    # answer_label = "<s>"+ style + ',' + label + "</s>"
    # sentence = form
    
    tokenized_data = tokenizer(sentence, padding='max_length', max_length=1024, truncation=True)
    
    tokenized_label = tokenizer(answer_label, padding='max_length', max_length=512, truncation=True)
    
    ssm_encode_data_dict['input_ids'].append(tokenized_data['input_ids'])
    ssm_encode_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    
    ssm_decode_data_dict['input_ids'].append(tokenized_label['input_ids'])
    ssm_decode_data_dict['attention_mask'].append(tokenized_label['attention_mask'])
    
    

    return ssm_encode_data_dict, ssm_decode_data_dict


def get_dataset(data_path):
    raw_data = pd.read_csv(data_path)
    input_ids_list = []
    attention_mask_list = []

    decode_input_ids_list = []
    decode_attention_mask_list = []

    for utterance in raw_data.iterrows():
        ssm_encode_data_dict, ssm_decode_data_dict = tokenize_and_align_labels(utterance[1][1], utterance[1][2], utterance[1][3])
        input_ids_list.extend(ssm_encode_data_dict['input_ids'])
        attention_mask_list.extend(ssm_encode_data_dict['attention_mask'])

        decode_input_ids_list.extend(ssm_decode_data_dict['input_ids'])
        decode_attention_mask_list.extend(ssm_decode_data_dict['attention_mask'])


    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list), torch.tensor(decode_input_ids_list), torch.tensor(decode_attention_mask_list))
