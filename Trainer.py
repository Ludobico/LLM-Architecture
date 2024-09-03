import torch, os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from trl import SFTTrainer

from typing import Optional, Any

import random

class TrainQwen:
  def __init__(self) -> None:
    self.curdir = os.path.dirname(os.path.realpath(__file__))
    self.model_id = "Qwen/Qwen2-1.5B-Instruct"
    self.huggingface_cache_dir = "/data/HuggingFace"
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    self.output_dir = "/data/LLM_Checkpoint"
    self.epoch = 3
    self.output_name = f"Qwen1.5b_SFT_{self.epoch}ep"

  
  @classmethod
  def dataset_loader(cls, dataset_id : str = 'maywell/ko_wikidata_QA') -> DatasetDict:
    """
    ```python
    DatasetDict({
    train: Dataset({
        features: ['instruction', 'output'],
        num_rows: 137505
      })
    })
    ```
    """
    dataset = load_dataset(dataset_id, cache_dir=cls().huggingface_cache_dir)

    return dataset
  
  @classmethod
  def chat_template(cls):
    """
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is the Qwen2?<|im_end|>
    <|im_start|>assistant
    Qwen2 is the new series of Qwen large language models<|im_end|>
    <|im_start|>user
    Tell me more<|im_end|>
    <|im_start|>assistant
    """
    tokenizer = AutoTokenizer.from_pretrained(cls().model_id, cache_dir = cls().huggingface_cache_dir)

    chat = [
      {"role" : "user", "content" : "What is the Qwen2?"},
      {"role" : "assistant", "content" : "Qwen2 is the new series of Qwen large language models"},
      {"role" : "user", "content" : "Tell me more"}
    ]

    encoded = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    print(encoded)
  
  @staticmethod
  def dataset_preprocessor(dataset : Any, question : str, answer : str):
    data = dataset.map(lambda x : {'text' : f"<|im_start|>user\n{x[question]}<|im_end|>\n<|im_start|>assistant\n{x[answer]}<|im_end|>"})
    return data

  @staticmethod
  def dataset_validation(dataset, count : int = 5, random_index : bool = True) -> None:
    dataset_length = len(dataset['train'])
    for i in range(count):
      random_index = random.randint(0, dataset_length -1)
      print('-'*80)
      if random_index:
        print(dataset['train'][random_index]['text'])
      else:
        print(dataset['train'][i]['text'])
      print('-'*80)
  
  @classmethod
  def do_train(cls):
    torch.cuda.empty_cache()
    device = cls().device
    model_id = cls().model_id

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map = device, cache_dir = cls().huggingface_cache_dir, torch_dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = cls().huggingface_cache_dir)

    dataset = TrainQwen.dataset_loader()
    processed_data = TrainQwen.dataset_preprocessor(dataset=dataset, question="instruction", answer="output")

    TrainQwen.dataset_validation(processed_data, count=5)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    trainer = SFTTrainer(
      model=model,
      train_dataset=processed_data['train'],
      dataset_text_field='text',
      args=TrainingArguments(
        output_dir=os.path.join(cls().output_dir, cls().output_name),
        per_device_train_batch_size=8,
        num_train_epochs=cls().epoch,
        learning_rate=1e-5,
        logging_steps=20,
        optim="paged_adamw_8bit",
        save_total_limit=cls().epoch,
        save_strategy="steps",
        save_steps=0.1
      ),
      data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
  TrainQwen.do_train()