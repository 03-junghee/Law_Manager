import os
import json
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# 모델 및 토크나이저 로드
model = BertModel.from_pretrained("./trained_model")
tokenizer = BertTokenizer.from_pretrained("./trained_model")

# 전처리된 데이터를 불러오는 함수 (재사용)
def load_preprocessed_data_for_inference(data_dir):
    cases = []
    file_names = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    case_data = json.load(file)
                    text = case_data.get("판시사항", "") + " " + case_data.get("판결요지", "") + " " + case_data.get("이유", "")
                    cases.append((text, case_data))  # (텍스트, 전체 데이터) 튜플로 저장
                    file_names.append(file_name)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"파일 {file_name}에서 에러 발생: {e}")
    
    return cases, file_names

# 유사 판례를 찾는 함수
def find_most_similar_case(input_text, model, tokenizer, cases, file_names):
    # 모델을 평가 모드로 전환
    model.eval()
    
    # 입력 텍스트 벡터화
    input_encoding = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        input_vector = model(**input_encoding).last_hidden_state[:, 0, :]  # CLS 토큰 사용

    similarities = []
    for case_text, _ in cases:
        case_encoding = tokenizer(case_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            case_vector = model(**case_encoding).last_hidden_state[:, 0, :]  # CLS 토큰 사용
        
        # 코사인 유사도 계산
        similarity = F.cosine_similarity(input_vector, case_vector)
        similarities.append(similarity.item())

    # 가장 높은 유사도를 가진 판례 찾기
    most_similar_idx = torch.tensor(similarities).argmax().item()
    return file_names[most_similar_idx], similarities[most_similar_idx], cases[most_similar_idx][1]  # 전체 데이터 반환

# 실행 예시
data_dir = 'processed_cases'
input_text = "어떤 사람이 헌법적 권리에 대해 이의를 제기하고 있습니다. 표현의 자유와 관련된 사건입니다."

# 전처리된 케이스 로드
cases, file_names = load_preprocessed_data_for_inference(data_dir)

# 유사한 판례 찾기
similar_case, similarity_score, case_data = find_most_similar_case(input_text, model, tokenizer, cases, file_names)
print(f"가장 유사한 판례: {similar_case}")
print(f"유사도 점수: {similarity_score}")
print("판례 내용:")
print(json.dumps(case_data, ensure_ascii=False, indent=4))
