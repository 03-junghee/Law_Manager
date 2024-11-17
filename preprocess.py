import re
import os
import json
import pdfplumber

# PDF 파일에서 텍스트를 추출하는 함수 (pdfplumber 사용)
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"파일 {file_path} 처리 중 오류 발생: {e}")
    return text

# 텍스트에서 주요 정보를 추출하는 함수
def extract_case_info(text):
    case_info = {}
    case_info['판시사항'] = extract_section(text, r"【판시사항】", r"【판결요지】")
    case_info['판결요지'] = extract_section(text, r"【판결요지】", r"【참조조문】")
    case_info['주문'] = extract_section(text, r"【주문】", r"【이유】")
    case_info['이유'] = extract_section(text, r"【이유】", r"【결론】", end_at_last=True)
    return clean_case_data(case_info)

# 특정 섹션을 추출하는 함수
def extract_section(text, start_marker, end_marker, end_at_last=False):
    pattern = f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}" if not end_at_last else f"{re.escape(start_marker)}(.*)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else "정보 없음"

# 데이터 정제 함수
def clean_case_data(case_data):
    cleaned_data = {}
    for key, value in case_data.items():
        cleaned_value = re.sub(r'\s+', ' ', value).strip()
        cleaned_data[key] = cleaned_value
    return cleaned_data

# 폴더 내 모든 PDF 파일을 처리하고 JSON 파일로 저장하는 함수
def process_pdfs_in_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"처리 중: {file_name}")
            
            # PDF에서 텍스트 추출 및 정보 추출
            text = extract_text_from_pdf(file_path)
            case_info = extract_case_info(text)
            
            # JSON 파일명 설정 및 저장
            base_name = os.path.splitext(file_name)[0]
            output_json_path = os.path.join(output_folder, f"{base_name}.json")
            save_case_info_to_json(case_info, output_json_path)

            print(f"{file_name}의 정보가 {output_json_path}에 저장되었습니다.")

# 결과 저장 함수
def save_case_info_to_json(case_info, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(case_info, f, ensure_ascii=False, indent=4)

# 실행 예시
input_folder_path = 'train_data'  # PDF 파일이 들어 있는 폴더 경로
output_folder_path = 'processed_cases'  # 출력 JSON 파일이 저장될 폴더 경로

# 폴더 내 모든 PDF 파일 처리
process_pdfs_in_folder(input_folder_path, output_folder_path)
print("모든 PDF 파일 처리가 완료되었습니다.")
