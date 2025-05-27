from .model import tokenizer, model, device
import torch
import torch.nn.functional as F
import emoji

def detect_emotion_from_text(text: str) -> dict:
    clean_text = emoji.replace_emoji(text, replace='')
    inputs = tokenizer(clean_text, return_tensors='pt', truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Get the emotion with highest probability
    emotion_idx = probs.argmax()
    emotion_label = model.config.id2label[emotion_idx]
    emotion_score = float(probs[emotion_idx])

    return {
        'label': emotion_label,
        'score': emotion_score
    }
