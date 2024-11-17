import os
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# 커스텀 데이터셋 클래스
class LegalCaseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # labels의 dtype을 torch.float으로 설정
        return item

    def __len__(self):
        return len(self.labels)



# 전처리된 데이터를 불러오는 함수
def load_preprocessed_data(data_dir):
    texts = []
    labels = []
    file_names = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                case_data = json.load(file)
                text = case_data.get("판시사항", "") + " " + case_data.get("판결요지", "") + " " + case_data.get("이유", "")
                texts.append(text)
                labels.append(0)  # 임시 레이블 (분류 문제가 아닌 유사도 기반 검색이므로 필요 없음)
                file_names.append(file_name)
    
    return texts, labels, file_names

# 모델 학습 함수
def train_model(train_texts):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # 토큰화
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    dataset = LegalCaseDataset(encodings, [0] * len(train_texts))  # 임시 레이블

    # 학습 파라미터 설정
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # 모델 학습 시작
    trainer.train()

    # 학습된 모델 저장
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

    return model, tokenizer

# 학습 실행
data_dir = 'processed_cases'  # 전처리된 JSON 파일들이 있는 폴더
train_texts, _, _ = load_preprocessed_data(data_dir)
train_model(train_texts)
