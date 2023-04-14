# CM-KoBERT train과 test를 위한 코드
# input : train.tsv, test.tsv, mfcc.npy (preprocessing.py 실행 후 생성됨.)
# output : cmbert_pred.npy (모델 예측 라벨), cmbert_truth.npy (실제 라벨), cmbert_output.npy (모델 예측 값)

from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE,cached_path
from model import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from utils import *

from sklearn.metrics import f1_score, accuracy_score

from kobert_hf.kobert_tokenizer import KoBERTTokenizer

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(i):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='data/', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='kobert-base-v1/', type=str, 
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default='Multi', type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default='CM-BERT_output/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.'store_true'")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=24, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=24, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size", default=24, type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.5e-5")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=11111,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    processors = {
        "multi": PgProcessor,
    }

    num_labels_task = {
        "multi": 7,
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    seed_num = np.random.randint(1,10000)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_num)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    
    tokenizer = KoBERTTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case) ### KoBERT

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs


    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format("-1"))

    model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels = num_labels)

    for name, param in model.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.0" in name or "encoder.layer.1" in name:
            param.requires_grad = True
        if "encoder.layer.2" in name or "encoder.layer.3" in name :
            param.requires_grad = True
        if "encoder.layer.4" in name or  "encoder.layer.5" in name:
            param.requires_grad = True
        if "encoder.layer.6" in name or "encoder.layer.7" in name:
            param.requires_grad = True
        if "encoder.layer.8" in name or "encoder.layer.9" in name :
            param.requires_grad = True
        if "encoder.layer.10" in name or  "encoder.layer.11" in name:
            param.requires_grad = True
        if "BertFinetun" in name or "pooler" in name:
            param.requires_grad = True

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_decay = ['BertFine']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(np in n for np in new_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay )and any(np in n for np in new_decay)],'lr':0.01}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    audio_npy = np.load('./data/audio/mfcc.npy')
    train_audio, test_audio = audio_npy[:8231], audio_npy[8231:]

    if args.do_train:

        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_train_audio = torch.tensor(train_audio, dtype=torch.float32)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float32)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_train_audio, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, all_train_audio, label_ids = batch
                loss = model(input_ids, all_train_audio,segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args.do_test:

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("")
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)

        all_test_audio = torch.tensor(test_audio, dtype=torch.float32,requires_grad=True)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float32)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_test_audio)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

        model.eval()

        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        predict_list = []
        truth_list = []
        text_attention_list = []
        fusion_attention_list = []

        logits_list = []

        with torch.no_grad():
            for input_ids, input_mask, segment_ids, label_ids, all_test_audio in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                all_test_audio = all_test_audio.to(device)

                with torch.no_grad():
                    tmp_test_loss = model(input_ids, all_test_audio,segment_ids, input_mask, label_ids)
                    logits, text_attention, fusion_attention = model(input_ids, all_test_audio, segment_ids, input_mask)

                logits_tmp = logits.detach().cpu().numpy()

                logits_list.append(logits_tmp)
                
                logits = torch.argmax(logits, dim=1)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                text_attention = text_attention.cpu().numpy()
                fusion_attention = fusion_attention.cpu().numpy()
                test_loss += tmp_test_loss.mean().item()

                for i in range(len(logits)):
                    predict_list.append(logits[i])
                    truth_list.append(label_ids[i])
                    text_attention_list.append(text_attention[i])
                    fusion_attention_list.append(fusion_attention[i])
                nb_test_examples += input_ids.size(0)
                nb_test_steps += 1

        cmbert_output_tmp = np.array(logits_list)
        #np.save('cmbert_output_tmp.npy', cmbert_output_tmp)

        cmbert_output = []
        for i in range(len(cmbert_output_tmp)):
            cmbert_output.extend(cmbert_output_tmp[i])
        np.save('cmbert_output.npy', cmbert_output)

        predict_list = np.array(predict_list).reshape(-1)
        truth_list = np.array(truth_list)

        np.save('cmbert_pred.npy', predict_list)
        np.save('cmbert_truth.npy', truth_list)

        f_score = f1_score(truth_list, predict_list, average='weighted')
        acc = accuracy_score(predict_list, truth_list)

        results = {'accuracy':acc, 'F1 score':f_score}

        return results

if __name__ == "__main__":
    results = main(1)
    print(results)