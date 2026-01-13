import os
import json
import pandas as pd
from tqdm import tqdm
import re
import requests
import argparse

# -----------------------------
# Configuration: API Key & URL
# -----------------------------
api_key = os.environ.get("AIMLAPI_KEY", "YOUR_API_KEY_HERE")
base_url = os.environ.get("AIMLAPI_URL", "https://api.aimlapi.com/v1")

CHAT_URL = f"{base_url}/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# -----------------------------
# Helper Functions
# -----------------------------
def detect_question_type(question):
    # Ensure question is a string to prevent errors with empty/NaN Excel cells
    q = str(question).lower() 
    if "select all that apply" in q:
        return "multiple"
    elif "select one option" in q:
        return "single"
    elif "true or false" in q:
        return "bool"
    else:
        return "short"

def clean_gpt_response(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```", "", content)
    content = re.sub(r"```$", "", content)
    return content.strip()

def gpt_judge_match(question, answer, prediction, model_name="gpt-4o"):
    qtype = detect_question_type(question)
    if qtype == "short":
        type_instruction = (
            "- This is a short answer/open-ended question. Extract the final answer only. "
            "Judge leniently: accept semantically or medically equivalent answers, "
            "even if wording, format, or minor details differ."
        )
    else:
        type_instruction = {
            "multiple": (
                "- Multiple-choice (select all that apply). Extract final answer and match all options, ignore order."
            ),
            "single": (
                "- Single-choice. Extract final answer and match by letter or content."
            ),
            "bool": (
                "- True/False. Extract final answer and match boolean meaning leniently."
            ),
        }[qtype]

    prompt = f"""You are a medical evaluation expert.

1. Extract the final answer only from the model's prediction below.
2. Judge if it matches the provided ground-truth answer.

{type_instruction}

Return ONLY a JSON object like:

{{
  "match": true or false,
  "final_answer": "the extracted final answer text"
}}

Question:
{question}

Ground-truth Answer:
{answer}

Model's Prediction:
{prediction}
"""

    try:
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a medical evaluation expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }

        response = requests.post(CHAT_URL, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        content = clean_gpt_response(content)
        parsed = json.loads(content)
        return parsed.get("match", False), parsed.get("final_answer", "").strip()
    except Exception as e:
        print("❌ ERROR", e)
        return None, None

# -----------------------------
# Data Loading (Excel)
# -----------------------------
def load_excel(excel_path):
    try:
        # Default to using openpyxl engine
        df = pd.read_excel(excel_path, engine='openpyxl')
        
        # Check if standard columns exist
        if "question" in df.columns and "answer" in df.columns:
            return df, True
        else:
            # If standard columns are missing, try reading without header
            df = pd.read_excel(excel_path, header=None, engine='openpyxl')
            return df, False
    except Exception as e:
        print("❌ Failure:", e)
        raise e

# -----------------------------
# Evaluation Logic
# -----------------------------
def evaluate(json_path, excel_path, output_path=None, model_name="gpt-4o"):
    print(f"Loading predictions from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)
    
    print(f"Loading ground truth from: {excel_path}")
    gt_df, has_header = load_excel(excel_path)

    results = []
    total, correct = 0, 0

    print(f"Starting evaluation using model: {model_name}")
    for item in tqdm(prediction_data):
        index = int(item["index"])
        prediction = item.get("prediction", "").strip()
        if not prediction:
            continue
        
        # Check Excel boundaries (Excel rows often 1-based, DataFrame 0-based)
        if (index - 1) >= len(gt_df):
            print(f"⚠️ Index {index} out of range in Excel file.")
            continue
            
        if has_header:
            question = str(gt_df.iloc[index-1]["question"])
            answer = str(gt_df.iloc[index-1]["answer"])
        else:
            question = str(gt_df.iloc[index-1][0])
            answer = str(gt_df.iloc[index-1][1])

        match, final_answer = gpt_judge_match(question, answer, prediction, model_name)
        if match is None:
            continue

        symbol = "✔" if match else "✘"
        status = "✅" if match else "❌"
        print(f"[{symbol}] index: {index} | Final Answer: {final_answer} | Groundtruth: {answer} | Match: {status}")

        results.append({
            "index": index,
            "match": match,
            "final_answer": final_answer,
            "groundtruth": answer,
            "question": question,
            "prediction": prediction
        })

        total += 1
        if match:
            correct += 1

    # Calculate final statistics
    accuracy = correct / total if total > 0 else 0.0

    print(f"\n✅ Judged {total} predictions")
    print(f"✅ Correct: {correct}")
    print(f"✅ Accuracy: {accuracy:.2%}")

    if output_path:
        # Create a structure that holds both summary stats and detailed results
        final_output = {
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "accuracy_display": f"{accuracy:.2%}",
                "judge_model": model_name
            },
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"✅ Results with accuracy saved to: {output_path}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical QA predictions against Excel ground truth.")
    
    parser.add_argument("--json_path", type=str, required=True, help="Path to the prediction JSON file.")
    parser.add_argument("--excel_path", type=str, required=True, help="Path to the ground truth Excel file.")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", help="Path to save the evaluation results JSON.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name to use for evaluation (default: gpt-4o).")

    args = parser.parse_args()

    evaluate(
        json_path=args.json_path,
        excel_path=args.excel_path,
        output_path=args.output_path,
        model_name=args.model
    )