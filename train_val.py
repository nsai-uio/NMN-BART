import sys, os
import json
import pickle
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from transformers import AutoTokenizer, Seq2SeqTrainingArguments,AutoConfig
import torch
import logging
from evaluate import load
import nltk
import numpy as np
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from dataset_util import *
from typing import List, Optional, Tuple
from utils.utils import DataCollatorForNmnBart
from dataset_util import find_vision_feat, most_frequent
torch.autograd.set_detect_anomaly(True)
from eval_metrics.cider.cider import Cider
from eval_metrics.spice.spice import Spice

question_file = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
answer_file = 'data/v2_mscoco_train2014_annotations.json'
feat_file = "data/trainval_feature.h5"
explanation_file = 'data/textual/train_exp_anno.json'

question_file_val = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
answer_file_val = 'data/v2_mscoco_val2014_annotations.json'
explanation_file_val = 'data/textual/val_exp_anno.json'


def file2dataset(q_file, an_file, ex_file, feat_file, split):
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
        #print(question_set[0])
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
        
        # parse all dataset
        k = 0
        for q_id, ex in explanation_dict.items():
            k += 1
            q_index = q_id_to_q_index[int(q_id)]
            q_set = question_set[q_index]
    
            questions.append(q_set['question'])
            img_ids.append(q_set['image_id'])
            
            explanations.append(ex)
            an_idx = question_id_to_answer_index[int(q_id)]
            answers.append(annotations[an_idx]['answers'])
            v, r = find_vision_feat(feat_file, q_set['image_id'], feat_coco_id_to_index, use_spatial)
            vision_features.append(v)
            relation_masks.append(r)
            if k > 10 and split == 'train':
                break
            if k > 1000 and split == 'val':
                break


    file_dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'explanation': explanations,
        'img_id': img_ids,
        'vision_features': vision_features,
        'relation_masks': relation_masks
    })
    return file_dataset

train_dataset = file2dataset(question_file, answer_file, explanation_file, feat_file, 'train')
val_dataset = file2dataset(question_file_val, answer_file_val, explanation_file_val, feat_file, 'val')


raw_datasets = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })

column_names = raw_datasets["train"].column_names
question_column = column_names[0]
explanation_column = column_names[2]
answer_column = column_names[1]

# facebook/bart-base, facebook/bart-large-cnn
bart_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-base"
    )

max_seq_length = 512  # tokenizer.model_max_length
padding = "max_length"
max_answer_length = 512
ignore_pad_token_for_loss = True

def preprocess_squad_batch(
    examples,
    question_column: str,
    explanation_column: str,
    answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    explanations = examples[explanation_column]
    answers = examples[answer_column]

    def generate_input(_question, _answer):
        return " ".join(["question:", _question.lstrip(), "answer:", _answer.lstrip()])

    def generate_output(_explanation):
        return " ".join(["explanation:", _explanation.lstrip()])

    inputs = [generate_input(question, answer) for question, answer in zip(questions, answers)]
    
    ex_str = ['. '.join(sublist) for sublist in explanations]
    targets = [generate_output(explanation) for explanation in ex_str]
    return inputs, targets

def preprocess_function(examples):
    inputs, targets = preprocess_squad_batch(examples, question_column, explanation_column, answer_column)

    model_inputs = bart_tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
    # Tokenize targets with text_target=...
    labels = bart_tokenizer(text=targets, max_length=max_answer_length, padding=padding, truncation=True)

    
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != bart_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(
    preprocess_function, 
    batched=True,
    # batch_size=2,
    # num_proc=16,
    # drop_last_batch=False,
    desc="Running tokenizer on train dataset",
    )

# evaluation
rouge = load("rouge")
bleu = load("bleu")
meteor = load("meteor")
cider = Cider()
spice = Spice()
nltk.download('punkt')

scorers = [
    (bleu, "BLEU"),
    (meteor, "METEOR"),
    (rouge, "ROUGE"),
    (cider, "CIDEr"),
    (spice, "SPICE"),
    ]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = bart_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, bart_tokenizer.pad_token_id)
    decoded_labels = bart_tokenizer.batch_decode(labels, skip_special_tokens=True)
    eval_results = {}
    for scorer, method in scorers:
        print('computing %s score...' % (method))
        if method == "ROUGE":
            # Rouge expects a newline after each sentence
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
            # Note that other metrics may not have a `use_aggregator` parameter
            # and thus will return a list, computing a metric for each sentence.
            rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
            # Extract a few results
            rouge_result = {key: value * 100 for key, value in rouge_result.items()}
            # Add mean generated length
            prediction_lens = [np.count_nonzero(pred != bart_tokenizer.pad_token_id) for pred in predictions]
            rouge_result["gen_len"] = np.mean(prediction_lens)
            eval_results['rouge'] = {k: round(v, 4) for k, v in rouge_result.items()}
        if method == "BLEU":
            bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
            bleu_result['bleu'] = round(bleu_result['bleu'], 4)
            bleu_result['precisions'] = [round(v, 4) for v in bleu_result['precisions']]
            eval_results['bleu'] = {k: bleu_result[k] for k in ['bleu', 'precisions']}
        if method == "METEOR":
            meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            eval_results['meteor'] = {k: round(v, 4) for k, v in meteor_result.items()}
        if method == "CIDEr":
            cider_result, _ = cider.compute_score(decoded_preds, decoded_labels)
            eval_results["cider"] = round(cider_result, 4)
        if method == "SPICE":
            try:
                spice_result = spice.compute_score(decoded_preds, decoded_labels)
            except:
                eval_results["spice"] = 0
            else:
                eval_results["spice"] = round(spice_result, 4)
    return eval_results

batch_size = 24 #16 # 10
args = Seq2SeqTrainingArguments(
    f"facebook/bart-base",   
    evaluation_strategy = "epoch",
    learning_rate=2e-5,  
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=15,
    predict_with_generate=True,
    fp16=True,
    max_grad_norm=1.0
)

from model.nmn_bart_deep import BartForNLE
config = AutoConfig.from_pretrained("facebook/bart-base")
# facebook/bart-base, facebook/bart-large-cnn

module_kwargs = {
    'dim_v': 512, # node embedding
    'dim_hidden': config.d_model,
    'dim_edge': 256,
    'dim_vision': 2053, # 2048 + 5 spatial
    'dropout_prob': 0.5,
    'T_ctrl': 3,
    'glimpses': 2,
    # 'device': device,
    'stack_len': 4,
    'use_gumbel': 1,
    'use_validity': 1,
    }

model = BartForNLE(config, module_kwargs)
# config the generate sequence length
model.config.max_length = 512
model.config.min_length = 56
model.config.output_hidden_states = True

data_collator = DataCollatorForNmnBart(
    bart_tokenizer, 
    model=model,
    )

from vqa_nmn_bart_trainer import NmnBartTrainer
trainer = NmnBartTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=bart_tokenizer, 
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("tmp/model_deep")