import os
import json

cases_path = "data/cases_for_analysis.json"

arabic_cases_list = []

with open(cases_path, "r", encoding="utf-8") as f:
    cases = json.load(f)

for i, case_json in enumerate(cases):
    case_dict_sample = {}
    
    try:    
        case_dict_sample["case_origin_path"] = f"case_{i}"
        case_dict_sample["facts"] = case_json["structured_case"]["facts_steps"]
        case_dict_sample["steps_texts"] = case_json["structured_case"]["reasoning_steps"]
        case_dict_sample["verdict_summary"] = case_json["structured_case"]["verdict_summary"]
        
        case_dict_sample["laws"] = [k+" : "+v for k,v in case_json["laws_desc"].items()]
        case_dict_sample["verdict_classification"] = case_json["classification"]["verdict_classification"]["ruling_side"]
        
        arabic_cases_list.append(case_dict_sample)
        
    except Exception as e:
        print(f"Error at case_{i}: {e}")

with open("data/arabic_analysis_cases.json", "w") as f:
    json.dump(arabic_cases_list, f, ensure_ascii=False)

   
english_cases_list = []

for i, case_json in enumerate(cases):
    case_dict_sample = {}

    try:    
        case_dict_sample["case_origin_path"] = f"case_{i}"
        case_dict_sample["facts"] = case_json["structured_case"]["fact_steps_english"]
        case_dict_sample["steps_texts"] = case_json["structured_case"]["reason_steps_english"]
        case_dict_sample["verdict_summary"] = case_json["structured_case"]["verdict_english"]
        
        case_dict_sample["laws"] = [k+" : "+v for k,v in case_json["laws_desc_eng_translation"].items()]
        case_dict_sample["verdict_classification"] = case_json["classification"]["verdict_classification"]["ruling_side"]
        
        english_cases_list.append(case_dict_sample)
        
    except Exception as e:
        print(f"Error at case_{i}: {e}")

with open("data/english_analysis_cases.json", "w") as f:
    json.dump(english_cases_list, f, ensure_ascii=False)    
