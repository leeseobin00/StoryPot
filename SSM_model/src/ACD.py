import torch


from utils import jsonlload, parse_args,jsondump
from googletrans import Translator


import copy
import os

from ACD_models import ACD_model
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from rouge import Rouge
from ACD_dataset import label_id_to_name, get_dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

task_name = 'ABSA' 

label_id_to_name = ['True', 'False']
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import wandb





def predict_from_korean_form(tokenizer, CD_model,entity_property_dev_dataloader,a,name):
    wandb.init(project="grad", entity="leoyeon", name = f'{name}')

    style_correct = 0 
    rouge = Rouge()
    rouge1_f = 0
    rouge2_f = 0
    rougel_f = 0
    for batch in entity_property_dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            e_input_ids, e_input_mask, d_input_ids,d_attention_mask = batch
            pred_list = []
            label_list = []
            with torch.no_grad():
                predictions = CD_model.test(e_input_ids, e_input_mask)
                pred_list.extend(tokenizer.batch_decode(predictions, skip_special_tokens=True))
                label_list.extend(tokenizer.batch_decode(d_input_ids, skip_special_tokens=True))
                

            pred_num = 0  
            

            for true,pred in zip(label_list,pred_list):
                test1 = true
                test2 = pred

                style1 = test1.split(',',maxsplit=1)
                style2 = test2.split(',',maxsplit=1)
                docs1 = style1[1]
                docs2 = style2[1]

                if style1[0] == 'black and white':
                    if style2[0] == 'sketch':
                        style_correct = style_correct+1
                elif style1[0] == 'sketch':
                    if style2[0] == 'black and white':
                        style_correct = style_correct+1
                elif style1[0] == 'fantasy vivid colors':
                    if style2[0] == 'Oil Paintings':
                        style_correct = style_correct+1
                elif style1[0] == 'Oil Paintings':
                    if style2[0] == 'fantasy vivid colors':
                        style_correct = style_correct+1
                elif style1[0] == 'vivid color':
                    if style2[0] == 'Oil Paintings':
                        style_correct = style_correct+1
                elif style1[0] == 'Oil Paintings':
                    if style2[0] == 'vivid color':
                        style_correct = style_correct+1
                elif style1[0] == 'Oil Paintings':
                    if style2[0] == 'oriental painting':
                        style_correct = style_correct+1        
                elif style1[0] == 'oriental painting':
                    if style2[0] == 'Oil Paintings':
                        style_correct = style_correct+1

                if style1[0] == style2[0]:
                    style_correct = style_correct+1     
                                    

                scores = rouge.get_scores(docs1, docs2, avg=True)
                rouge1_f = rouge1_f + scores['rouge-1']['f']
                rouge2_f = rouge2_f + scores['rouge-2']['f']
                rougel_f = rougel_f + scores['rouge-l']['f']
                        
    print(style_correct,pred_num)
    wandb.log({"style_Match": style_correct})
    wandb.log({"style_accuracy": style_correct/a})
    wandb.log({"rouge1_f1score": rouge1_f/a})
    wandb.log({"rouge2_f1score": rouge2_f/a})
    wandb.log({"rougel_f1score": rougel_f/a})
   
    
        
def test_sentiment_analysis(args):
    
    #inference할 모델을 불러오는 경로
    
    entity_property_dev_data = get_dataset("validation.csv")
    entity_property_dev_dataloader = DataLoader(entity_property_dev_data, shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=8,
                                pin_memory=True,)
    tsvPth_test = "/home/ksw2/hdd2/GraduationWork/T5_summarization/val.csv"
    
    #토크나이저를 불러와서 스페셜 토큰 형태로 추가
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
    a = len(entity_property_dev_data)
     
    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/Summarize/saved_model/category_extraction/adddata3/saved_model_epoch_2.pt"        
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()   
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1,entity_property_dev_dataloader,a,'t5_saved_model_2.pt')

    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/Summarize/saved_model/category_extraction/adddata3/saved_model_epoch_3.pt"        
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()   
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1,entity_property_dev_dataloader,a,'kobart_saved_model_epoch_11.pt')

    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/류상연/saved_model_epoch_16.pt"        
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()   
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1,entity_property_dev_dataloader,a,'kobart_saved_model_epoch_12.pt')

    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/류상연/saved_model_epoch_20.pt"        
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()   
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1,entity_property_dev_dataloader,a,'kobart_saved_model_epoch_13.pt')

    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/Summarize/saved_model/category_extraction/adddata3/saved_model_epoch_17.pt"        
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()   
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1,entity_property_dev_dataloader,a,'kobart_saved_model_epoch_15.pt')

    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/Summarize/saved_model/category_extraction/adddata3/saved_model_epoch_18.pt"        
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()   
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1,entity_property_dev_dataloader,a,'kobart_saved_model_epoch_17.pt')

    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/Summarize/saved_model/category_extraction/adddata3/saved_model_epoch_16.pt"        
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()   
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1,entity_property_dev_dataloader,a,'kobart_saved_model_epoch_19.pt')

    


if __name__ == "__main__":
    args = parse_args()

    test_sentiment_analysis(args)
    