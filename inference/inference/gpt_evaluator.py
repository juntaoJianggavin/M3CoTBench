import os
import json
from typing import Dict, Any, Optional, List
import time
from loguru import logger as eval_logger

def get_chat_response(client, prompt, model=None, max_token=10000, retry=5):
    """
    Get response from the GPT model.
    
    Args:
        client: OpenAI client instance.
        prompt: The input prompt.
        model: Model name.
        max_token: Maximum number of tokens.
        retry: Number of retries.
        
    Returns:
        The GPT response.
    """
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    # If no model is specified, attempt to get it from environment variables
    if model is None:
        model = os.getenv("MODEL_VERSION", "gpt-4o")
    
    for i in range(retry):
        try:
            completion = client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=0.5 * i, 
                max_tokens=max_token
            )
            prediction = completion.choices[0].message.content.strip()
            if prediction.lower() == "yes" or prediction.lower() == "no":
                return prediction
        except Exception as e:
            eval_logger.error(f"Attempt {i+1} failed: {e}")
            time.sleep(1)  # Pause briefly before retrying
    
    return "no"  # Default return "no"

def build_evaluation_prompt(question_data):
    """
    Build the evaluation prompt.
    
    Args:
        question_data: Dictionary containing the question, answer, and prediction.
        
    Returns:
        The evaluation prompt.
    """
    prompt = """You are given a question, the solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "Yes" if the solution is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question: 
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution: 
{solution}
--------------------------------
"""
    question = question_data["question"]
    answer = question_data["answer"]
    response = str(question_data["response"])
    prompt = prompt.format(question=question, answer=answer, solution=response)
    return prompt

def evaluate_prediction(question_data, client, model=None):
    """
    Evaluate the correctness of the prediction using GPT.
    
    Args:
        question_data: Dictionary containing the question, answer, and prediction.
        client: OpenAI client instance.
        model: Model name.
        
    Returns:
        Evaluation result: 1 indicates correct, 0 indicates incorrect.
    """
    # Build evaluation prompt
    prompt = build_evaluation_prompt(question_data)
    
    try:
        # Get GPT evaluation result
        completion = get_chat_response(client, prompt, model)
        
        if completion.lower() == "yes" or completion.lower() == "no":
            judge_result = 1 if completion.lower() == "yes" else 0
        else:
            eval_logger.error(f"Invalid response for index {question_data['index']}: {completion}")
            judge_result = 0
    except Exception as e:
        eval_logger.error(f"Error evaluating index {question_data['index']}: {e}")
        judge_result = 0
        
    return judge_result