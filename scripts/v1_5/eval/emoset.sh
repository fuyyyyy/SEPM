#!/bin/bash

MODEL=llava-v1.5-7b

python3 -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$MODEL \
    --image-folder /data0/EmoSet/image \
    --stage1_question_file /data0/EmoSet/question/stage_1.jsonl \
    --stage1_answers_file /data0/EmoSet/answer/stage_1.jsonl \
    --stage2_question_file /data0/EmoSet/question/stage_2.jsonl \
    --stage2_answers_file /data0/EmoSet/answer/stage_2.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
