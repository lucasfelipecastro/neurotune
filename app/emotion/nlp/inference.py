from .model import tokenizer, model, device
import torch, torch.nn.functional as F
import emoji

GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def detect_emotion_from_text(text: str) -> dict:
    clean_text = emoji.replace_emoji(text, replace='')
    inputs = tokenizer(clean_text, return_tensors='pt', truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.sigmoid(logits)[0].cpu().numpy()

    # Catch emotions with probability greater than 0.3 (adjustable threshold)
    emotion_scores = {
        label: float(prob)
        for label, prob in zip(GOEMOTIONS_LABELS, probs)
        if prob > 0.3
    }

    # Sort by score
    sorted_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True))

    return sorted_emotions
