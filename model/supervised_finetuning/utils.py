from pathlib import Path

import yaml
from custom_datasets import QA_SPECIAL_TOKENS, get_one_dataset
from custom_datasets.dialogue_collator import DialogueDataCollator
from losses import CrossEntropyLoss, PolyLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import nltk
from functools import partial
import numpy as np

def get_tokenizer(conf):
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name, cache_dir=conf.cache_dir)

    if "galactica" in conf.model_name:
        tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>"})
    elif "GPT-JT" in conf.model_name:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token, "sep_token": "<|extratoken_100|>"})
    elif "codegen" in conf.model_name:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>", "sep_token": "<|endoftext|>"})

    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
    additional_special_tokens = list(set(additional_special_tokens + list(QA_SPECIAL_TOKENS.values())))

    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    return tokenizer

# placeholder for now
def preprocess_qa(eval_pred):
    return (eval_pred.predictions, eval_pred.label_ids)

def postprocess_summarization(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def preprocess_summarization(eval_pred, tokenizer, ignore_pad_token_for_loss=True):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_summarization(decoded_preds, decoded_labels)
    return decoded_preds, decoded_labels

 
def get_metrics(conf, tokenizer):
    # to be extended or updated
    qa_datasets = ["squad_v2", "adversarial_qa", "trivia_qa_context", "trivia_qa_noconext"]
    summarization_datasets = ["xsum", "cnn_dailymail", "samsum", "multi_news"]

    # the reason behind using a list is that we might want to extend the list of our
    # metrics in the future for more thorough evaluation
    if any(dataset in qa_datasets for dataset in conf.datasets):
        metrics, preprocess_fn = [evaluate.load("squad_v2")], preprocess_qa
    elif any(dataset in summarization_datasets for dataset in conf.datasets):
        metrics, preprocess_summarization = [evaluate.load("rouge")], partial(preprocess_summarization, tokenizer, 
                                                                             ignore_pad_token_for_loss=conf.ignore_pad_token_for_loss)
    else:
        raise ValueError("Unknown dataset / task")
    return metrics

def get_model(conf, tokenizer):
    # encoder-decoder support for Flan-T5 like models
    # for now, we can use an argument but in the future,
    # we can automate this
    if conf.seq2seqmodel:
        model = AutoModelForSeq2SeqLM.from_pretrained(conf.model_name, cache_dir=conf.cache_dir)
    else:     
        model = AutoModelForCausalLM.from_pretrained(conf.model_name, cache_dir=conf.cache_dir)

    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        assert not conf.freeze_layer, "Cannot change the number of embeddings if the model is frozen."

    model.resize_token_embeddings(len(tokenizer))

    if conf.freeze_layer:
        model = freeze_top_n_layers(model, conf.freeze_layer)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    return model


def get_dataset(conf, tokenizer):
    train_datasets, evals = [], {}

    for dataset_name in conf.datasets:
        train, val = get_one_dataset(conf, dataset_name)
        train_datasets.append(train)
        evals[dataset_name] = Subset(val, list(range(min(len(val), conf.eval_size)))) if conf.eval_size else val

    train = ConcatDataset(train_datasets)

    collate_fn = DialogueDataCollator(tokenizer, max_length=conf.max_length)

    return train, evals, collate_fn


def get_loss(loss, poly_eps):
    if loss == "CrossEntropyLoss":
        return CrossEntropyLoss()
    elif loss == "Poly":
        return PolyLoss(epsilon=poly_eps)
    else:
        raise ValueError(f"Loss {loss} not supported")


def read_yamls(dir):
    conf = {}
    no_conf = True

    for config_file in Path(dir).glob("**/*.yaml"):
        no_conf = False
        with config_file.open("r") as f:
            conf.update(yaml.safe_load(f))

    if no_conf:
        print(f"WARNING: No yaml files found in {dir}")

    return conf


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=666, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def freeze_top_n_layers(model, target_layers):
    # its possible we can simply detect which module is a ModuleList
    # and simply freeze the module without doing string parsing
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad = False
        elif ".layer" in name or ".h." in name:
            tokens = name.split(".")
            layer_ = None
            for token in tokens:
                if token.isdigit():
                    layer_ = int(token)
                    break

            if layer_ is not None and layer_ < target_layers:
                # print('freeze ', layer_, name)
                param.requires_grad = False
    return model


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("bigscience/bloomz-560m")
    freeze_top_n_layers(model, 10)
    print(model.state_dict().keys())
