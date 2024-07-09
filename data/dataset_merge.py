from datasets import load_dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict

class DatasetCollection:
    def __init__(self) -> None:
        self.huggingface_cache_dir = "/data/HuggingFace"
    
    @classmethod
    def dataset_open_orca(cls, dataset_id : str = "kyujinpy/KOR-OpenOrca-Platypus-v3") -> DatasetDict:
        dataset = load_dataset(dataset_id, cache_dir=cls().huggingface_cache_dir)
        return dataset

    @classmethod
    def dataset_koalpaca(cls, dataset_id : str = "beomi/KoAlpaca-v1.1a")-> DatasetDict:
        dataset = load_dataset(dataset_id, cache_dir=cls().huggingface_cache_dir)
        return dataset

    @classmethod
    def dataset_kowiki(cls, dataset_id : str = "maywell/ko_wikidata_QA")-> DatasetDict:
        dataset = load_dataset(dataset_id, cache_dir=cls().huggingface_cache_dir)
        return dataset
    
    @staticmethod
    def rename_columns(dataset : DatasetDict, question_col : str, answer_col : str) -> DatasetDict:
        if question_col not in dataset.column_names:
            dataset = dataset.rename_column(question_col, 'instruction')
        if answer_col not in dataset.column_names:
            dataset = dataset.rename_column(answer_col, 'output')
        dataset = dataset.select_columns(['instruction', 'output'])
        return dataset

    @classmethod
    def merge_datasets(cls):
        open_orca_dataset = cls.dataset_open_orca()
        koalpaca_dataset = cls.dataset_koalpaca()
        kowiki_dataset = cls.dataset_kowiki()

        open_orca_dataset = cls.rename_columns(open_orca_dataset['train'], "instruction", "output")
        koalpaca_dataset = cls.rename_columns(koalpaca_dataset['train'], "instruction", "output")
        kowiki_dataset = cls.rename_columns(kowiki_dataset['train'], "instruction", "output")

        merged_train_dataset = concatenate_datasets([
            open_orca_dataset, koalpaca_dataset, kowiki_dataset
        ])

        merged_dataset = DatasetDict({
            'train' : merged_train_dataset
        })

        return merged_dataset