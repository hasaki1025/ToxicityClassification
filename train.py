import torch.optim
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

model_name = "bert-base-uncased"  # 可以根据需要选择不同的BERT模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分类




def train(model,train_data_iter,device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    sum_loss = 0.0
    nums_batch = 0
    for batch in tqdm(train_data_iter):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.mean().backward()
        optimizer.step()
        with torch.no_grad():
            sum_loss += loss.item()
            nums_batch += 1

    return sum_loss/nums_batch

def evaluate(model,eval_data_iter,device):
    model.eval()
    for batch in tqdm(eval_data_iter):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels).logits

