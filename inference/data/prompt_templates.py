from typing import Dict, Any, Optional, List, Union
import string
import pandas as pd

def get_cot_prompt(question: str, options: Optional[Dict[str, str]] = None) -> str:
    """
    Get Chain-of-Thought (CoT) reasoning prompt template.
    
    Args:
        question: Question text.
        options: Dictionary of options (e.g., {'A': 'Option A', 'B': 'Option B'}).
        
    Returns:
        Formatted prompt.
    """
    prompt = question.strip()
    
    # Add options (if any)
    if options and len(options) > 0:
        prompt += "\n" + "\n".join([f"{key}. {item}" for key, item in options.items()])
    
    # Add CoT instruction
    prompt += "\nPlease generate a step by step answer, include all your intermediate reasoning process, and provide the final answer at the end."
    
    return prompt

def get_direct_prompt(question: str, options: Optional[Dict[str, str]] = None) -> str:
    """
    Get direct reasoning prompt template.
    
    Args:
        question: Question text.
        options: Dictionary of options (e.g., {'A': 'Option A', 'B': 'Option B'}).
        
    Returns:
        Formatted prompt.
    """
    prompt = question.strip()
    
    # Add options (if any)
    if options and len(options) > 0:
        prompt += "\n" + "\n".join([f"{key}. {item}" for key, item in options.items()])
    
    # Add direct answer instruction
    prompt += "\nPlease directly provide the final answer without any other output."
    
    return prompt

def get_system_prompt(model_type: str = "general") -> str:
    """
    Get the system prompt

    Args:
        model_type: Model type, 'general' or 'medical'

    Returns:
        System prompt
    """
    if model_type == "medical":
        return (
            "You are an experienced medical expert specializing in analyzing medical images and answering related questions. "
            "You will carefully examine all details in the images and use your medical knowledge to provide accurate analyses and answers."
        )
    else:
        return (
            "You are a vision-language assistant capable of analyzing medical images and answering related questions. "
            "Please provide accurate and helpful answers based on the image content."
        )

def extract_options_from_doc(doc: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract options from a document.
    
    Args:
        doc: Document data.
        
    Returns:
        Dictionary of options.
    """
    return {cand: doc[cand] for cand in string.ascii_uppercase if cand in doc and not pd.isna(doc[cand])}

def build_gpt4_evaluation_prompt(question_data: Dict[str, Any]) -> str:
    """
    Build prompt for GPT-4 evaluation.
    
    Args:
        question_data: Dictionary containing question, answer, and solution.
        
    Returns:
        Formatted prompt.
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
    return prompt.format(question=question, answer=answer, solution=response)