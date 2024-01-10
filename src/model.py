from transformers import AutoModel

def CodeLlamaForSequenceClassification(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model

# Example usage
model_name = "bert-base-uncased"
model = CodeLlamaForSequenceClassification(model_name)

# You can now use the model for further processing
