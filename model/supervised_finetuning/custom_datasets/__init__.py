from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from .prompt_dialogue import PromptGeneratedDataset

QA_SPECIAL_TOKENS = {"Question": "<question>", "Answer": "<answer>"}
SUMMARIZATION_SPECIAL_TOKENS = {"Text" : "", "Summary" : "TL;DR:"}

summarization_name_mapping = {
    "cnn_dailymail": ("article", "highlights"),
    "samsum": ("dialogue", "summary"),
    "xsum": ("document", "summary"),
    "multi_news": ("document", "summary"),
}


class QADataset(Dataset):
    def __init__(self, dataset, cache_dir, split):
        self.dataset = load_dataset(dataset, cache_dir=cache_dir, split=split)
        # handle conversion between the dataset formats if needed by creating a
        # preprocess function here
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # return first answer form list of possible answers
        return data["title"] + ". " + data["context"] + " " + data["question"], data["answers"]["text"][0]


class SummarizationDataset(Dataset):
    def __init__(self, dataset, cache_dir, split):
        self.dataset = load_dataset(dataset, cache_dir=cache_dir, split=split)
        self.summary_column, self.text_column = summarization_name_mapping[dataset]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        # dummy return first answer
        return " ".join(SUMMARIZATION_SPECIAL_TOKENS["Text"], data[self.text_column], " ", 
                        SUMMARIZATION_SPECIAL_TOKENS["Summary"], data[self.summary_column])
        

class WebGPT(Dataset):
    def __init__(self) -> None:
        super().__init__()

        dataset = load_dataset("openai/webgpt_comparisons")
        questions = {}
        # using prompt as our index will allows us
        # to add additional generated prompt later
        self.index2question = {}
        for row in dataset["train"]:
            question = row["question"]["full_text"]
            if question not in self.index2question:
                self.index2question[len(self.index2question)] = question

            # only keep the best answer
            questions[question] = row["answer_0" if row["score_0"] > row["score_1"] else "answer_1"]

        self.questions = questions

    def __len__(self):
        return len(self.index2question)

    def __getitem__(self, index):
        question = self.index2question[index]
        answer = self.questions[question]
        return [question, answer]


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=666, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_one_dataset(conf, dataset_name):
    dataset_name = dataset_name.lower()

<<<<<<< HEAD
    if dataset_name in ["squad_v2", "adversarial_qa", "trivia_qa_context", "trivia_qa_noconext"]:
        train = QADataset(dataset_name, conf.cache_dir, "train")
        eval = QADataset(dataset_name, conf.cache_dir, "validation")
    
    elif dataset_name in ["xsum", "cnn_dailymail", "samsum", "multi_news"]:
        train = SummarizationDataset(dataset_name, conf.cache_dir, "train")
        eval = SummarizationDataset(dataset_name, conf.cache_dir, "validation")        
        
=======
    if dataset_name == "squadv2":
        train = SquadV2Dataset(conf.cache_dir, "train")
        eval = SquadV2Dataset(conf.cache_dir, "validation")
>>>>>>> 66891dd690d86f341c76bd249a89f7c2235ffe00
    elif dataset_name == "webgpt":
        dataset = WebGPT()
        train, eval = train_val_dataset(dataset, val_split=0.2)
    elif dataset_name == "prompt_dialogue":
        dataset = PromptGeneratedDataset()
        train, eval = train_val_dataset(dataset, val_split=0.2)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return train, eval
