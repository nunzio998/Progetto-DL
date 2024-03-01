import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import re
from tqdm import tqdm  # Libreria per vedere avanzamento processo

# Load del dataset
from datasets import load_dataset

ds = load_dataset("big_patent")

# Visualizzazione dataset
print(ds)
# print(ds["test"]["description"][1])
# print("\n" + ds["test"]["abstract"][1])

# Split del dataset

x_train_text = ds['train']['description']
y_train = ds['train']['abstract']

x_validation_text = ds['validation']['description']
y_validation = ds['validation']['abstract']

x_test_text = ds['test']['description']
y_test = ds['test']['abstract']

#print("-----------\n" + x_test_text[1])


# PREPROCESSING DATI
# Trattamento dei dati tramite regex

def noise_remover(text):
    text2 = re.sub('<[^>]*>', '', text)  # remove html tags
    text2 = re.sub('[^\w\s\d]', ' ', text2)  # remove punctuations by negating chars, spaces, and digits
    text2 = re.sub('\s+', ' ', text2)  # remove consecutive spaces
    return text2


# Rimozione rumore
# Utilizzo tqdm per monitorare l'avanzamento
x_train_text_processed = noise_remover(x_train_text)
x_validation_text_processed = noise_remover(x_validation_text)
x_test_text_processed = noise_remover(x_train_text)

#print("-----------\n" + x_test_text[1])
