import torch
from transformers import (RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, AdamW,
                          get_linear_schedule_with_warmup,XLMRobertaForSequenceClassification, XLMRobertaTokenizer
)
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse
from utils import TextDataset
from model import Classifier
import json
import logging
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from utils import TextDataset, set_seed, draw_plot, get_date, makedir

logger = logging.getLogger(__name__)


def load_data_from_csv(csv_data):
    csv_data = csv_data[['instruction', 'label']]
    data_list = []
    for instru, label in csv_data.values:
        data_list.append([instru, label])
    return data_list



def convert_to_encoding(input_data , target, tokenizer, max_token_length):
    
   
    func_after_encoding = tokenizer(input_data, return_tensors="pt", padding="max_length", max_length=max_token_length,
                                    truncation=True)
    label = target.view(-1, 1).float()
    return func_after_encoding, label


def train(model, train_data_loader, val_data , tokenizer, device, args):
    ymd, hms = get_date() 
    
    checkpoint_path = os.path.join(args.output_dir, 'best-acc-checkpoint', ymd + '-' + hms)
    if args.pretraining_model_path != '':  
        pretrained_model_name = args.pretraining_model_path.split('/')[-4]
        pretrained_method_name = args.pretraining_model_path.split('/')[-4]
        tensorboard_path = os.path.join(args.output_dir, 'tbd-logs','use-pretraining', pretrained_method_name, ymd, hms)
    else:  
        pretrained_model_name = ''
        tensorboard_path = os.path.join(args.output_dir, 'tbd-logs', 'no-pretraining', ymd, hms)
    logger.info("Best accuracy checkpoint path created in: %s", checkpoint_path)
    path_dict = {'best_acc': checkpoint_path}
    makedir(path_dict)
    writer = SummaryWriter(log_dir=str(tensorboard_path))
    args_dict = vars(args)
    dataset_path = args.dataset
    dataset_path = dataset_path.split('/')[-2]
    
    with open(os.path.join(str(checkpoint_path), 'args.json'), 'w') as f:
        json.dump(args_dict, f)
    
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(args.num_epochs * len(train_data_loader)) * 0.1,
                                                num_training_steps=(args.num_epochs * len(train_data_loader)))

    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1_score = 0.0
    model.to(device)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_data_loader, mininterval=10)
        for idx , batch in enumerate(pbar):
            
            input_data , target = batch
            input_encoding, label = convert_to_encoding(input_data , target, tokenizer,
                                                        args.max_token_length)  
            input_encoding_gpu = input_encoding.to(device)
            label_gpu = label.to(device)
            loss, _ = model(input_encoding=input_encoding_gpu, label=label_gpu)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            train_steps += 1
            pbar.set_description(
                f"epoch {epoch} loss {round(total_loss / train_steps, 5)}") if train_steps % 10 == 0 else None
        logger.info(f"Epoch {epoch} is finished, the average loss is {total_loss / train_steps}")
        eval_params = evaluate(model, device, tokenizer, val_data , args)
        if eval_params['accuracy'] > best_accuracy:
            
            best_accuracy = eval_params['accuracy']
        if eval_params['precision'] > best_precision:
            best_precision = eval_params['precision']
        if eval_params['recall'] > best_recall:
            best_recall = eval_params['recall']
        if eval_params['f1_score'] > best_f1_score:
            best_f1_score = eval_params['f1_score']
        writer.add_scalar('/Evaluation/Accuracy', eval_params['accuracy'], epoch)
        writer.add_scalar('/Evaluation/Precision', eval_params['precision'], epoch)
        writer.add_scalar('/Evaluation/Recall', eval_params['recall'], epoch)
        writer.add_scalar('/Evaluation/F1 Score', eval_params['f1_score'], epoch)
        writer.add_scalar('/Evaluation/Avg Loss', eval_params['avg_loss'], epoch)
        writer.add_scalar('/Train/Avg Loss', total_loss / train_steps, epoch)
    writer.add_hparams(
        {'max_token_length': args.max_token_length, 'lr': args.learning_rate, 'bsize': args.batch_size,
         'epochs': args.num_epochs, 'pretrained_model': pretrained_model_name, 'seed': args.seed,'dataset':dataset_path,},
        {
            'best_accuracy': best_accuracy,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_f1_score': best_f1_score,
            
        })
    writer.close()



def evaluate(model, device, tokenizer, val_data , args):
    eval_data_list = load_data_from_csv(val_data)
    eval_dataset = TextDataset(eval_data_list)
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_data_loader = DataLoader(eval_dataset, args.batch_size, sampler=eval_sampler)
    model.eval()
   
    cm_all_predicted = []
    cm_all_labels = []
    eval_loss = 0.0
    eval_steps = 0
    with torch.no_grad():
        for  input_data , target in eval_data_loader:
            input_encoding, label = convert_to_encoding(input_data,target, tokenizer,
                                                        args.max_token_length)  
            input_encoding_gpu = input_encoding.to(device)
            label_gpu = label.to(device)
            loss, output = model(input_encoding=input_encoding_gpu, label=label_gpu)
            custom_threshold = 0.5
            predicted = output >= custom_threshold
            
            cm_all_predicted.extend(predicted.int().cpu().numpy())
            cm_all_labels.extend(label_gpu.int().cpu().numpy())
            eval_loss += loss.item()
            eval_steps += 1
    avg_loss = eval_loss / eval_steps
    conf_matrix = confusion_matrix(cm_all_labels, cm_all_predicted)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)  
    precision_pos = tp / (tp + fp)  
    precision_neg = tn / (tn + fn)  
    recall_pos = tp / (tp + fn)  
    recall_neg = tn / (tn + fp)  
    f1_score_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)  
    f1_score_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)  
    logger.info(f'Confusion Matrix:: {conf_matrix}')
    logger.info(f'Accuracy: {accuracy * 100:.4f}%')
    logger.info(f'Positive Precision: {precision_pos * 100:.4f}%')
    logger.info(f'Negative Precision: {precision_neg * 100:.4f}%')
    logger.info(f'Positive Recall: {recall_pos * 100:.4f}%')
    logger.info(f'Negative Recall: {recall_neg * 100:.4f}%')
    logger.info(f'Positive F1 Score: {f1_score_pos:.4f}')
    logger.info(f'Negative F1 Score: {f1_score_neg:.4f}')
    return {'accuracy': accuracy, 'precision': precision_pos, 'recall': recall_pos, 'f1_score': f1_score_pos,
            'avg_loss': avg_loss}



def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretraining_model_path", default='', type=str, required=False,
                        help="contrastive learned model")
    parser.add_argument("--learning_rate", default='2e-5', type=float, required=True,
                        help="classifier learned rate")
    parser.add_argument("--max_token_length", default=256, type=int, required=False,
                        help="the max number of token length")
    parser.add_argument("--batch_size", default=4, type=int, required=False,
                        help="the number of batch size")
    parser.add_argument("--num_epochs", default=10, type=int, required=False,
                        help="the number of epochs to train")
    parser.add_argument("--seed", default=42, type=int, required=True,
                        help="the number of seed to use")
    parser.add_argument("--gpu", default=0, type=int, required=False,
                        help="the number of gpu to use")
    parser.add_argument("--dataset", default='', type=str, required=True,
                        help="the dataset of classifier to use")
    
    args = parser.parse_args()
    data_path = args.dataset
    data = pd.read_csv(data_path)

    
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)

    args.output_dir = 'output/classifier'
    args.dataset = data_path
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("args: %s", args)
    set_seed(args.seed)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    
    config.num_labels = 1
    if args.pretraining_model_path != '': 
        classifier_encoder = RobertaForSequenceClassification.from_pretrained(args.pretraining_model_path,
                                                                              config=config)

    else: 
        classifier_encoder =RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config)
    classifier_model = Classifier(classifier_encoder)
    train_dataset = load_data_from_csv(train_data)
    train_dataset = TextDataset(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler)
    train(classifier_model, train_data_loader, val_data , tokenizer, device, args)


if __name__ == '__main__':
    main()

