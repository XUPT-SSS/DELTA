from transformers import (RobertaTokenizer, RobertaModel, DataCollatorForLanguageModeling, AdamW,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, RandomSampler
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from model import ContrastiveLearning
import argparse
from utils import TextDataset, set_seed, get_date, makedir
import logging
import os , json
logger = logging.getLogger(__name__)



def load_data_from_csv(file_path):
    csv_data = pd.read_csv(file_path, low_memory=False,
                           usecols=['anchor_data','neg_data'])  
    data_list = []
    for anchor,neg_data in tqdm(csv_data.values, desc='load data:'):
        # before = ' '.join(code_before.split())
        # after = ' '.join(code_after.split())
        data_list.append({'anchor':anchor ,'neg_data': neg_data})
    return data_list



def convert_to_encoding(batch, tokenizer, args):
    code_list = []
    for anchor,neg_data in zip(batch['anchor'], batch['neg_data']):
        code_list.append(anchor)
        # code_list.append(pos_data)
        code_list.append(neg_data)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    code_token = tokenizer(code_list, return_tensors="pt", padding="max_length",
                           max_length=args.max_token_length,
                           truncation=True, return_special_tokens_mask=True)

    if args.do_mlm:
        code_encoding = data_collator([code_token])
   
        code_encoding['input_ids'] = code_encoding['input_ids'][0]
        
        code_encoding['attention_mask'] = code_encoding['attention_mask'][0]
        
        code_encoding['labels'] = code_encoding['labels'][0]
        return code_encoding
    else:
        code_encoding = code_token
        return code_encoding



def train(model, train_data_loader, tokenizer, device, args):

    ymd, hms = get_date() 
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint/')
    if args.do_contrastive:
        checkpoint_path = str(checkpoint_path) + 'Contra'
    if args.do_mlm:
        checkpoint_path = str(checkpoint_path) + 'MLM'
    checkpoint_path = os.path.join(str(checkpoint_path), ymd + '-' + hms)
    logger.info("checkpoint_path created in :" + str(checkpoint_path))
    path_dict = {'last_epoch': os.path.join(str(checkpoint_path), 'last_epoch'),
                 'best_loss': os.path.join(str(checkpoint_path), 'best_loss')}
    makedir(path_dict)
    
    args_dict = vars(args)

    
    with open(os.path.join(str(checkpoint_path), 'args.json'), 'w') as f:
        json.dump(args_dict, f)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(args.num_epochs * len(train_data_loader)) * 0.1,
                                                num_training_steps=(args.num_epochs * len(train_data_loader)))
    if args.checkpoint_dir != '':
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'CL_checkpoint.pth'))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        args.start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    model.train()
    best_loss = float('inf')
    for epoch in range(args.start_epoch, args.num_epochs):
        total_loss, total_contra_loss, total_mlm_loss = 0.0, 0.0, 0.0
        train_step = 0
        loss = None
        pbar = tqdm(train_data_loader, mininterval=10)
        for batch in pbar:
            input_encoding = convert_to_encoding(batch, tokenizer, args).to(device)
            contra_loss, masked_lm_loss = model(input_encoding, args)
            loss = contra_loss + masked_lm_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            total_contra_loss += contra_loss.item()
            total_mlm_loss += masked_lm_loss.item()
            train_step += 1
            pbar.set_description(f"epoch {epoch} total Loss: {round(total_loss / train_step, 5)}, "
                                 f"cl_loss: {round(total_contra_loss / train_step, 5)}, "
                                 f"mlm_loss: {round(total_mlm_loss / train_step, 5)}") if train_step % 20 == 0 else None
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, path_dict['last_epoch'] + '/CL_checkpoint.pth')

        if total_loss < best_loss:  
            model.encoder.save_pretrained(path_dict['best_loss'])
            best_loss = total_loss
        if (epoch + 1) % 20 == 0: 
            model.encoder.save_pretrained(checkpoint_path + '/CL_model_epoch_' + str(epoch))
        logger.info(f"Epoch {epoch} of {args.num_epochs} is finished. "
                    f"Total Loss: {total_loss / train_step}, "
                    f"cl_loss: {total_contra_loss / train_step}, "
                    f"mlm_loss: {total_mlm_loss / train_step}")


def evaluate(model, data_loader, optimizer, num_epochs):
    model.eval()
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filename", default='', type=str, required=True,
                        help="train data file name")
    parser.add_argument("--test_filename", default='', type=str, required=False,
                        help="test data file name")
    parser.add_argument("--max_token_length", default=256, type=int, required=False,
                        help="the max number of token length")
    parser.add_argument("--batch_size", default=4, type=int, required=False,
                        help="the number of batch size")
    parser.add_argument("--num_epochs", default=10, type=int, required=False,
                        help="the number of epochs to train")
    parser.add_argument("--gpu", default=0, type=int, required=False,
                        help="the number of gpu to use")
    parser.add_argument("--learning_rate", default='5e-5', type=float, required=True,
                        help="contrastive learned rate")
    parser.add_argument("--temperature", default='0.1', type=float, required=True,
                        help="temperature for contrastive loss")
    parser.add_argument("--do_contrastive", action='store_true',
                        help="Whether to run Contrastive learning.")
    parser.add_argument("--do_mlm", action='store_true',
                        help="Whether to run mask language model.")
    parser.add_argument("--checkpoint_dir", default='', type=str, required=False,
                        help="Resuming training.")

    args = parser.parse_args()

    args.output_dir = 'output/pretraining'
    args.start_epoch = 0


    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("args: %s", args)
    set_seed()

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    if args.checkpoint_dir != '':
        code_bert_model = RobertaModel.from_pretrained(args.checkpoint_dir)
    else:
        code_bert_model = RobertaModel.from_pretrained("microsoft/codebert-base")

    code_bert_model.to(device)

   
    csv_file_path = args.train_filename
    train_data_list = load_data_from_csv(csv_file_path)
    train_dataset = TextDataset(train_data_list)
    train_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler)
    contrastive_model = ContrastiveLearning(code_bert_model, device)
    train(contrastive_model, train_data_loader, tokenizer, device, args)


# 主函数
if __name__ == "__main__":
    main()
