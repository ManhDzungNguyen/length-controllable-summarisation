import os
import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from custom_dataset import LazyCustomDataset


def process_data(train_file, validation_file, tokenizer):
    train_dataset = LazyCustomDataset(train_file, tokenizer)
    eval_dataset = LazyCustomDataset(validation_file, tokenizer)

    return train_dataset, eval_dataset


def get_model(pretrained_model="google/flan-t5-small", tokenizer="google/flan-t5-small"):
    config = AutoConfig.from_pretrained(pretrained_model, dropout=0.3, use_cache=False)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[SN]", "[SEP]"]})

    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    model.resize_token_embeddings(len(tokenizer))

    assert (
        len(tokenizer) == model.get_input_embeddings().weight.shape[0]
    ), "Tokenizer vocabulary size does not match model input embedding size"
    return model, tokenizer

def train(pretrained_model, tokenizer, train_file, validation_file, saved_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(pretrained_model, tokenizer)

    train_dataset, eval_dataset = process_data(train_file, validation_file, tokenizer)

    model.to(device)
    print(device)

    training_args = TrainingArguments(
        output_dir=saved_dir,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=5,
        load_best_model_at_end=True,
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        gradient_accumulation_steps=16,
        warmup_steps=20,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    working_dir = "/home/kuuhaku/work/length-controllable-summarisation/en"
    data_dir = os.path.join(working_dir, "data/CNNDM/SentEnum/no_length_instruction")
    checkpoint_dir = os.path.join(working_dir, "models/sentenum_lenthprediction/checkpoints")

    train(
        pretrained_model="google/flan-t5-small",
        tokenizer = "google/flan-t5-small",
        train_file=os.path.join(data_dir, "train.json"),
        validation_file=os.path.join(data_dir, "validation.json"),
        saved_dir=checkpoint_dir
    )