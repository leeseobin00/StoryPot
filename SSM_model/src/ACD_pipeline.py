import os
import wandb
import torch
import torch.nn as nn

from ast import arg
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from transformers import AdamW
from tqdm import tqdm
from utils import set_seed, parse_args

from transformers import get_linear_schedule_with_warmup
from ACD_dataset import label_id_to_name, get_dataset
from datetime import datetime

from ACD_models import ACD_model
from rouge import Rouge


wandb.init(project="grad", entity="leoyeon")

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def train_sentiment_analysis(args):

    #Summarize Style Model CKPT 저장 위해 폴더 경로에 없을시 추가
    if not os.path.exists(args.SSM_model_path):
        os.makedirs(args.SSM_model_path)
        
    #랜덤시드 5로 고정
    random_seed_int = 5    
    set_seed(random_seed_int, device) 
    

    #데이터 로더 부분 Train 데이터셋 과 validation 데이터셋을 불러와 Dataloader에 넣고 배치 형태로 생성
    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-small-korean-summarization')

    train_data = get_dataset("data.csv")
    dev_data = get_dataset("Inference.csv")
    
    print('Summary_train_count: ', len(train_data))
    print('Summary_dev_count: ', len(dev_data))
    
    a = len(train_data)
    
    summary_train_dataloader = DataLoader(train_data, shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=8,
                                pin_memory=True,)
    summary_dev_dataloader = DataLoader(dev_data, shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=8,
                                pin_memory=True,)
    
    #모델 로딩 하고 GPU에 올림
    print('loading model')
    ssm_model = ACD_model(args, len(label_id_to_name), len(tokenizer))
    ssm_model.to(device)
    
    print('end loading')
    
    #옵티마이저 설정 부분 bias, LayerNrom으로 설정
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        ssm_param_optimizer = list(ssm_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        ssm_optimizer_grouped_parameters = [
            {'params': [p for n, p in ssm_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in ssm_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        ssm_param_optimizer = list(ssm_model.classifier.named_parameters())
        ssm_optimizer_grouped_parameters = [{"params": [p for n, p in ssm_param_optimizer]}]

    ssm_property_optimizer = AdamW(
        ssm_optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )
    #스케줄러 부분 현재 얼만큼 학습되고있는지 확인 할수 있음
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(summary_train_dataloader)

    ssm_scheduler = get_linear_schedule_with_warmup(
        ssm_property_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print('[Training Phase]')
    epoch_step = 0
    best_acc= -1
    
    #모델 학습 부분
    for _ in tqdm(range(epochs), desc="Epoch"):
        epoch_step += 1

        ssm_model.train()

        # ssm_model_train
        ssm_total_loss = 0

        for step, batch in enumerate(tqdm(summary_train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            e_input_ids, e_input_mask, d_input_ids,d_attention_mask= batch
            
            labels = d_input_ids
            labels = torch.where(labels!=0,labels,-100)
        
        
            ssm_model.zero_grad()
            loss, logits = ssm_model(e_input_ids, e_input_mask, d_input_ids,d_attention_mask,labels)

            loss.backward()

            ssm_total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(parameters=ssm_model.parameters(), max_norm=max_grad_norm)
            ssm_property_optimizer.step()
            ssm_scheduler.step()

        avg_train_loss = ssm_total_loss / len(summary_train_dataloader)
        print("SSM_Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        model_saved_path = args.SSM_model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(ssm_model.state_dict(), model_saved_path)

        
        if args.do_eval:
            ssm_model.eval()

            pred_list = []
            label_list = []
            
            style_correct = 0 
            rouge = Rouge()
            rouge1_f = 0
            rouge2_f = 0
            rougel_f = 0
            
            for batch in summary_dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                e_input_ids, e_input_mask, d_input_ids,d_attention_mask = batch

                with torch.no_grad():
                    predictions = ssm_model.test(e_input_ids, e_input_mask)
                #print("예측",tokenizer.batch_decode(predictions, skip_special_tokens=True)) 
                #print("정답",tokenizer.batch_decode(d_input_ids, skip_special_tokens=True)) 
                pred_list.extend(tokenizer.batch_decode(predictions, skip_special_tokens=True))
                label_list.extend(tokenizer.batch_decode(d_input_ids, skip_special_tokens=True))
            
                #단순
                for true,pred in zip(label_list,pred_list):
                    test1 = true
                    test2 = pred

                    style1 = test1.split(',',maxsplit=1)
                    style2 = test2.split(',',maxsplit=1)
                    docs1 = style1[1]
                    docs2 = style2[1]

                    #
                    #Realism, Painterly art
                    #Oriental ink, Gray-scale
                    #Pop art, Cartoon
                    #Sketch, Carricature
                    #Black and white, Gray-scale
                    
                    if style1[0] == 'Black and white':
                        if style2[0] == 'Gray-scale':
                            style_correct = style_correct+1
                    elif style1[0] == 'Gray-scale':
                        if style2[0] == 'Black and white':
                            style_correct = style_correct+1
                    elif style1[0] == 'Realism':
                        if style2[0] == 'Painterly art':
                            style_correct = style_correct+1
                    elif style1[0] == 'Painterly art':
                        if style2[0] == 'Realism':
                            style_correct = style_correct+1
                    elif style1[0] == 'Oriental ink':
                        if style2[0] == 'Gray-scale':
                            style_correct = style_correct+1
                    elif style1[0] == 'Gray-scale':
                        if style2[0] == 'Oriental ink':
                            style_correct = style_correct+1
                    elif style1[0] == 'Pop art':
                        if style2[0] == 'Cartoon':
                            style_correct = style_correct+1        
                    elif style1[0] == 'Cartoon':
                        if style2[0] == 'Pop art':
                            style_correct = style_correct+1
                    elif style1[0] == 'Sketch':
                        if style2[0] == 'Carricature':
                            style_correct = style_correct+1        
                    elif style1[0] == 'Carricature':
                        if style2[0] == 'Sketch':
                            style_correct = style_correct+1

                    if style1[0] == style2[0]:
                        style_correct = style_correct+1     
                                        

                    scores = rouge.get_scores(docs1, docs2, avg=True)
                    rouge1_f = rouge1_f + scores['rouge-1']['f']
                    rouge2_f = rouge2_f + scores['rouge-2']['f']
                    rougel_f = rougel_f + scores['rouge-l']['f']

            print({"rouge1_f1score": rouge1_f/a})
            print({"rouge2_f1score": rouge2_f/a})
            print({"rougel_f1score": rougel_f/a})
            print({"style_Match": style_correct})
            print({"style_accuracy": style_correct/a})
            wandb.log({"style_Match": style_correct})
            wandb.log({"style_accuracy": style_correct/a})
            wandb.log({"rouge1_f1score": rouge1_f/a})
            wandb.log({"rouge2_f1score": rouge2_f/a})
            wandb.log({"rougel_f1score": rougel_f/a})
                    
            print("training is done")

    
if __name__ == "__main__":
    
    args = parse_args()

    train_sentiment_analysis(args)
    

    
