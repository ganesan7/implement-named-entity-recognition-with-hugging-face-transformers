import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
)

MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"

print("Loading NER model...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
model.eval()

def perform_ner(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [config.id2label[p] for p in predictions]

    results = []
    current_tokens = []
    current_label = None

    for token, label in zip(tokens[1:-1], labels[1:-1]):
        if token.startswith("##"):
            if current_tokens:
                current_tokens[-1] += token[2:]
            continue

        if label.startswith("B-") or label == "O":
            if current_tokens:
                results.append((" ".join(current_tokens), current_label))
                current_tokens = []
                current_label = None

            if label.startswith("B-"):
                current_tokens = [token]
                current_label = label[2:]

        elif label.startswith("I-"):
            entity_label = label[2:]
            if current_tokens and current_label == entity_label:
                current_tokens.append(token)
            else:
                if current_tokens:
                    results.append((" ".join(current_tokens), current_label))
                current_tokens = [token]
                current_label = entity_label

    if current_tokens:
        results.append((" ".join(current_tokens), current_label))

    return results

if __name__ == "__main__":
    text = """
    KDnuggets is a data science website near Boston, Massachusetts.
    Gregory Piatetsky-Shapiro founded it.
    Microsoft is based in the United States.
    """

    print("\nNamed Entity Recognition Results:", flush=True)
    for entity, label in perform_ner(text):
        print(f"{entity}: {label}")

    while True:
        text = input("\nEnter text (or type exit): ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            for entity, label in perform_ner(text):
                print(f"{entity}: {label}")
