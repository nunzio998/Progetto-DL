import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# Carica il tokenizer e il modello
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Preprocessa il testo
tokens = tokenizer.batch_encode_plus(texts, truncation=True, padding=True, return_tensors='pt')

# Crea il dataset PyTorch
dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)

# Split del dataset in training e test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader per training e test
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definisci il loss e l'ottimizzatore
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Addestra il modello
for epoch in range(epochs):
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Valuta il modello
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        # Calcola e stampa le metriche di valutazione
