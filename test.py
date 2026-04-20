import sys, os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, AutoConfig
from evaluate import load
import nltk
from datasets import Dataset
from dataset_util import *
from typing import List, Optional, Tuple
from utils.utils import DataCollatorForNmnBart
from vqa_nmn_bart_trainer import NmnBartTrainer
from model.nmn_bart_deep import BartForNLE
from eval_metrics.cider.cider import Cider
from eval_metrics.spice.spice import Spice
import argparse

question_file_val = 'data/v2_OpenEnded_mscoco_test2015_questions.json'
answer_file_val = 'data/v2_mscoco_val2014_annotations.json' # contains test data
explanation_file_val = 'data/textual/test_exp_anno.json'
feat_file = "data/test_feature.h5"

def file2dataset(q_file, an_file, ex_file, feat_file):
    # parse answers and index
    annotations = []
    for f in an_file.split(':'):
        annotations += json.load(open(f, 'r'))['annotations']
    question_id_to_answer_index = {anno['question_id']: i for i, anno in enumerate(annotations)}
    # update multiple answers to one answer
    for ann in annotations:
        answer_list = [_['answer'] for _ in ann['answers']]
        one_answer = most_frequent(answer_list)
        ann['answers'] = one_answer

    # vision feature and index
    with h5py.File(feat_file, 'r') as f_file:
        coco_ids = f_file['ids'][()]
    feat_coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
    use_spatial = True

    with open(q_file, 'r') as f:
        question_set = json.load(f)['questions']
    q_id_to_q_index = {que['question_id']: i for i, que in enumerate(question_set)}
        
    # explanation
    with open(ex_file, 'r') as f:
        explanation_dict = json.load(f)    
                
        questions = []
        answers = []
        img_ids = []
        explanations = []
        vision_features = []
        relation_masks = []
        
        for q_id, ex in explanation_dict.items():
            if int(q_id) not in q_id_to_q_index: continue
            q_index = q_id_to_q_index[int(q_id)]
            q_set = question_set[q_index]
    
            questions.append(q_set['question'])
            img_ids.append(q_set['image_id'])
            
            explanations.append(ex)
            if int(q_id) in question_id_to_answer_index:
                an_idx = question_id_to_answer_index[int(q_id)]
                answers.append(annotations[an_idx]['answers'])
            else:
                answers.append("") # Handle missing answers for test set if necessary
                
            v, r = find_vision_feat(feat_file, q_set['image_id'], feat_coco_id_to_index, use_spatial)
            vision_features.append(v)
            relation_masks.append(r)

    return Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'explanation': explanations,
        'img_id': img_ids,
        'vision_features': vision_features,
        'relation_masks': relation_masks
    })

bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

def preprocess_function(examples):
    questions = examples['question']
    explanations = examples['explanation']
    answers = examples['answer']
    inputs = [" ".join(["question:", q.lstrip(), "answer:", a.lstrip()]) for q, a in zip(questions, answers)]
    ex_str = ['. '.join(sublist) for sublist in explanations]
    targets = [" ".join(["explanation:", ex.lstrip()]) for ex in ex_str]

    model_inputs = bart_tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = bart_tokenizer(text=targets, max_length=512, padding="max_length", truncation=True)
    labels["input_ids"] = [[(l if l != bart_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    rouge = load("rouge"); bleu = load("bleu"); meteor = load("meteor")
    cider_scorer = Cider(); spice_scorer = Spice()
    predictions, labels = eval_pred
    decoded_preds = bart_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, bart_tokenizer.pad_token_id)
    decoded_labels = bart_tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    res = {}
    res['bleu'] = bleu.compute(predictions=decoded_preds, references=decoded_labels)['bleu']
    res['meteor'] = meteor.compute(predictions=decoded_preds, references=decoded_labels)['meteor']
    res['rouge'] = rouge.compute(predictions=decoded_preds, references=decoded_labels)['rougeL']
    res['cider'], _ = cider_scorer.compute_score(decoded_preds, decoded_labels)
    try: res['spice'] = spice_scorer.compute_score(decoded_preds, decoded_labels)
    except: res['spice'] = 0
    return {k: round(v, 4) for k, v in res.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved model directory')
    args_p = parser.parse_args()

    dataset = file2dataset(question_file_val, answer_file_val, explanation_file_val, feat_file)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    config = AutoConfig.from_pretrained("facebook/bart-base")
    module_kwargs = {
        'dim_v': 512, 'dim_hidden': config.d_model, 'dim_edge': 256, 'dim_vision': 2053,
        'dropout_prob': 0.5, 'T_ctrl': 3, 'glimpses': 2, 'stack_len': 4, 'use_gumbel': 1, 'use_validity': 1,
    }
    model = BartForNLE.from_pretrained(args_p.checkpoint, config=config, module_kwargs=module_kwargs)
    model.config.max_length = 512
    model.config.min_length = 56

    trainer = NmnBartTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(output_dir="tmp/test", predict_with_generate=True, fp16=True),
        data_collator=DataCollatorForNmnBart(bart_tokenizer, model=model),
        tokenizer=bart_tokenizer,
        compute_metrics=compute_metrics
    )

    print("Evaluating...")
    results = trainer.predict(tokenized_dataset)
    print(results.metrics)
    
    with open(os.path.join(args_p.checkpoint, "test_predictions.json"), "w") as f:
        json.dump(bart_tokenizer.batch_decode(results.predictions, skip_special_tokens=True), f, indent=4)