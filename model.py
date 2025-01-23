import torch.nn as nn
import torch
import torch.nn.functional as f



class Classifier(nn.Module):
    def __init__(self, encoder):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.criterion = nn.BCELoss()
        self.m = nn.Sigmoid()
        self.dropout = nn.Dropout(0)   

    def forward(self, input_encoding, label=None):
        output_encoding = self.encoder(input_ids=input_encoding['input_ids'], attention_mask=input_encoding['attention_mask'])[0]
        logits = self.dropout(output_encoding)
        prob = self.m(logits)
        if label is not None:  # train

            loss = self.criterion(prob, label)
            return loss, prob
        else:  # predict
            hidden_feature = self.fc(output_encoding.last_hidden_state[:, 0, :])
            c_output = self.m(self.mlp_layer(hidden_feature))
            return c_output


class ContrastiveLearning(nn.Module):
    def __init__(self, encoder, device):
        super(ContrastiveLearning, self).__init__()
        self.encoder = encoder
        # self.criterion = nn.CosineEmbeddingLoss(margin=0.2)  # 对比损失函数
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        self.device = device
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(768, 768).to(device),  
            nn.ReLU().to(device),
            nn.Linear(768, 768).to(device)  
        )
        self.config = encoder.config
        self.lm_head = RobertaLMHead(self.config).to(device)

    def forward(self, input_encoding, args):
        output_encoding = self.encoder(input_ids=input_encoding['input_ids'], attention_mask=input_encoding['attention_mask'])
        output_dropout_encoding = self.encoder(input_ids=input_encoding['input_ids'], attention_mask=input_encoding['attention_mask'])
        # output_neg_encoding = self.encoder(input_ids=input_encoding['input_ids'], attention_mask=input_encoding['attention_mask'])
        masked_lm_loss = torch.tensor(0.0, requires_grad=True)  # 初始化为0
        contra_loss = torch.tensor(0.0, requires_grad=True)   # 初始化为0
        if args.do_mlm:
            mlm_output = self.lm_head(output_encoding[0])
            masked_lm_loss = self.mlm_loss_fct(mlm_output.view(-1, self.config.vocab_size), input_encoding['labels'].view(-1))
            mlm_output_dropout = self.lm_head(output_dropout_encoding[0])
            masked_lm_loss += self.mlm_loss_fct(mlm_output_dropout.view(-1, self.config.vocab_size), input_encoding['labels'].view(-1))
            # mlm_neg = self.lm_head(output_encoding[0])
            # masked_lm_loss = self.mlm_loss_fct(mlm_neg.view(-1, self.config.vocab_size), input_encoding['labels'].view(-1))
            masked_lm_loss = masked_lm_loss / 2
        if args.do_contrastive:
            contra_output = self.mlp_layer(output_encoding.last_hidden_state[:, 0, :])
            contra_output_dropout = self.mlp_layer(output_dropout_encoding.last_hidden_state[:, 0, :])
            # contra_neg = self.mlp_layer(output_neg_encoding.last_hidden_state[:, 0, :])
            contra_loss_input = torch.stack((contra_output, contra_output_dropout), dim=1).view(-1, 768)   # 交替堆叠
            contra_loss = self.nt_xent_loss(contra_loss_input, temperature=args.temperature)
            # contra_loss = self.info_nce_loss(contra_loss_input, temperature=args.temperature)
        return contra_loss, masked_lm_loss

    def nt_xent_loss(self, x, temperature):
        # Cosine similarity
        xcs = f.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
        xcs[torch.eye(x.size(0)).bool()] = float("-inf")

        # Ground truth labels
        target = torch.arange(x.size(0)).to(self.device)
        target[0::2] += 1
        target[1::2] -= 1
        loss = f.cross_entropy(xcs / temperature, target, reduction="mean")
        # Standard cross-entropy loss
        return loss
    

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = f.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

