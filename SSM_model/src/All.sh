#!/bin/bash

#. conda/bin/activate
#conda activate [가상환경 이름:team#] #임시(lss)

echo run task: Train entity_property_model  !!
wandb login a118c97a05a319f5960e15c958fff2d8f6b1d8c3
python ACD_pipeline.py \
  --do_train \
  --do_eval \
  --learning_rate 3e-5 \
  --eps 1e-8 \
  --num_train_epochs 60 \
  --SSM_model_path ../saved_model/category_extraction/adddata4/ \
  --batch_size 4 \
  --classifier_hidden_size 768 \
