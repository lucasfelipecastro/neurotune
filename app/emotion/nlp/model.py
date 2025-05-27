from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = 'bhadresh-savani/roberta-base-go-emotions'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
