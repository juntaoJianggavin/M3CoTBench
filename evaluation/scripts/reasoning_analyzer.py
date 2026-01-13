#!/usr/bin/env python3
"""
Medical Reasoning Pathway Analyzer
Analyzes AI model responses to extract reasoning pathways into 4 categories:
modality, feature, conclusion, and others.

This module provides functions to analyze text using LLM APIs and extract
structured reasoning pathway information.
"""

import requests
import json
import time
import re
from typing import Optional
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# ============================================================
# Configuration Section - Please modify according to your needs
# ============================================================

# API Configuration
# Base URL for the LLM API (e.g., "https://api.example.com")
# The endpoint "/v1/chat/completions" will be appended automatically in the code
API_BASE_URL = "https://api.example.com"

# API Key (Please replace with your own API key)
API_KEY = "YOUR_API_KEY_HERE"

# Model Configuration
# Specify which model to use (e.g., "gpt-4o", "gemini-2.5-pro", "gpt-4")
# This is the model name used by your API service, not the model being analyzed
MODEL_NAME = "gpt-4o"  # Change this to your preferred model

# ============================================================
# Output Format Definition
# ============================================================

OUTPUT_FORMAT = """
{
  "modality_order": 1,
  "feature_order": 2,
  "conclusion_order": 3,
  "others_order": 4,
  "modality_subs": ["substring 1", "substring 2"],
  "feature_subs": ["substring 1", "substring 2"],
  "conclusion_subs": ["substring 1", "substring 2"],
  "others_subs": ["substring 1", "substring 2"]
}
"""

def get_system_prompt(question: Optional[str] = None):
    """Generate system prompt with optional question context"""
    return f"""You are a medical reasoning pathway analyzer. Analyze an AI's answer to a medical question by extracting information into 4 categories and determining their appearance order.

**Medical Question/Task:**
{question or "Not provided"}


## CATEGORIES

**1. modality_subs** - Imaging/examination methods
- WHAT is the image modality or examination types of the given image?
- Examples: "Fundus photography", "MRI (FLAIR)", "Anterior segment slit lamp examination", "CT scan", "Ultrasound B-scan", "Section stained with hematoxylin and eosin (H&E)", "microscopy", "ENT Endoscopy"

**2. feature_subs** -Characteristics visible in the image that obtained directly form the image
- WHAT can be directly seen or observed?
- Examples: "opaque lens with dense white structure", "ill-defined hyperintense lesion with heterogeneous internal signal in the right cerebral hemisphere", "mild sulcal effacement, no significant mass effect", "cross-sectional musculoskeletal structures with visible bone contours", "bright, dense structure visible in the right kidney's collecting system"

**3. conclusion_subs** - Key conclusions of diagnoses, recognition(cell/organ/instrument/types/surgical processes), anatomical locations, grading, counting
-  WHAT is the key conclusions of diagnoses, recognition(cell/organ/instrument/types/surgical processes), anatomical locations, grading, counting? The key conclusions should be related to the question.
- Examples: "mature cataract", "cataract", "consistent with low grade diffuse astrocytoma", "kidney stone", "left kidney", "Coats' disease", "Monocyte cell type", "The red area in the right image represents in the left image is the pubic symphysis", "The instrument in the image is a Tube", "The state of ultrasound probe is idle", "The cell number is xxx".

**4. others_subs** - Further explanations of the treatment info, action recommendations, symptoms, functions of a cell/organ/instrument, an other corresponding clinical knowledge,  analysis of the each option of the question
- WHAT explanations of the treatment info, action recommendations, symptoms, functions of a cell/organ/instrument clinical knowledge, option analysis of the question are mentioned?
- Examples: "Visual function assessment is needed", "Dehydration concentrates urine increasing supersaturation of stone-forming salts", "COVID-19 is caused by the SARS-CoV-2 virus", "For benign lesions, aggressive treatments like chemotherapy are usually unnecessary", "Option A: Dehydration increases risk. Option B: High oxalate intake promotes stones"

## EXTRACTION RULES

**For substrings (*_subs):**
- Extract COMPLETE phrases or sentences (not single words or fragments)
- Extract MULTIPLE substrings per category if present - each should be a distinct piece of information
- Copy EXACT text from original - do NOT paraphrase or summarize
- Use empty array [] if category has no content

**For order (*_order) - Identify the AI's reasoning pathway:**
- Scan text from START to END to determine which category appears first, second, third, fourth
- This reveals the AI's reasoning structure: does it go modality→feature→conclusion→others? Or feature→conclusion→modality? Or only uses knowledge (others) without other categories?
- Assign 1-4 based on appearance sequence:
  - First category that appears → order = 1
  - Second category that appears → order = 2
  - Third category that appears → order = 3
  - Fourth category that appears → order = 4
- If category does not appear in the text → order = 0
- Examples of reasoning patterns you might find:
  * (1,2,3,4): modality→feature→conclusion→others
  * (1,2,4,3): modality→feature→others→conclusion  
  * (2,3,1,0): feature→conclusion→modality (no others)
  * (0,0,0,1): only knowledge/option analysis (common in multiple choice explanations)

## EXAMPLE

**Input:**
"Fundus photography of the right eye shows an opaque lens with dense white structure. This appearance is consistent with mature cataract. Visual function assessment is needed."

**Output:**
```json
{{
  "modality_order": 1,
  "feature_order": 2,
  "conclusion_order": 3,
  "others_order": 4,
  "modality_subs": ["Fundus photography"],
  "feature_subs": ["opaque lens with dense white structure"],
  "conclusion_subs": ["mature cataract"],
  "others_subs": ["Visual function assessment is needed"]
}}


# CRITICAL: You must respond with ONLY valid JSON format. Do not include any other text before or after the JSON object.

# Your output must be valid JSON in this exact format:
# {OUTPUT_FORMAT}"""

def analyze_text_with_retry(text, question: Optional[str] = None, max_retries=3, delay=2, 
                           api_base_url: Optional[str] = None, api_key: Optional[str] = None,
                           model_name: Optional[str] = None):
    """
    Analyze text with retry mechanism
    
    Args:
        text (str): Text to analyze
        question (str, optional): Corresponding question for context
        max_retries (int): Maximum number of retry attempts
        delay (int): Delay between retries (seconds)
        api_base_url (str, optional): API base URL (defaults to global API_BASE_URL)
        api_key (str, optional): API key (defaults to global API_KEY)
        model_name (str, optional): Model name (defaults to global MODEL_NAME)
    
    Returns:
        dict: Analysis result with 'success' and 'data' fields
    """
    # Use provided parameters or fall back to global configuration
    base_url = api_base_url or API_BASE_URL
    api_key_to_use = api_key or API_KEY
    model_to_use = model_name or MODEL_NAME
    
    # Check if API key is configured
    if api_key_to_use == "YOUR_API_KEY_HERE":
        return {
            'success': False,
            'error': "API key not configured. Please set API_KEY in the configuration section."
        }
    
    # Generate system prompt based on whether question is provided
    current_system_prompt = get_system_prompt(question)
    
    for attempt in range(max_retries):
        try:
            print(f"{Fore.BLUE}🔄 Attempt {attempt + 1}/{max_retries}...")
            
            # Build request payload
            payload = {
                "model": model_to_use,
                "group": "auto",
                "messages": [
                    {
                        "role": "system",
                        "content": current_system_prompt
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            }

            # Set request headers and URL
            baseurl_clean = base_url.rstrip('/')
            url = baseurl_clean + "/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key_to_use}"
            }

            # Send request
            response = requests.request(
                "POST",
                url,
                data=json.dumps(payload),
                headers=headers,
                timeout=(10, 60)  # (connect timeout, read timeout)
            )
            response.raise_for_status()

            data = response.json()
            
            # Extract content
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content'].strip()
                
                # Try to parse JSON
                try:
                    # Clean possible markdown format
                    content = re.sub(r'```json\s*', '', content)
                    content = re.sub(r'```\s*$', '', content)
                    content = content.strip()
                    
                    # Try to fix common JSON issues
                    # 1. Find JSON object start and end positions
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        # Extract JSON portion
                        content = content[json_start:json_end+1]
                    
                    # 2. Try direct parsing first
                    try:
                        parsed_data = json.loads(content)
                    except json.JSONDecodeError as json_err:
                        # If failed, try to fix common JSON issues
                        print(f"{Fore.YELLOW}⚠️  Attempting to fix JSON format issues...")
                        
                        # Try to fix: remove possible control characters
                        content_fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
                        
                        # Try parsing again
                        try:
                            parsed_data = json.loads(content_fixed)
                        except json.JSONDecodeError:
                            # If still failed, save original content for debugging
                            print(f"{Fore.RED}❌ JSON fix failed")
                            print(f"{Fore.YELLOW}   Error location: {json_err}")
                            print(f"{Fore.YELLOW}   Original content first 500 chars: {content[:500]}")
                            if len(content) > 500:
                                print(f"{Fore.YELLOW}   Original content last 500 chars: {content[-500:]}")
                            raise json_err
                    
                    # Validate required fields (must match OUTPUT_FORMAT)
                    required_fields = ['modality_order', 'feature_order', 'conclusion_order', 'others_order',
                                     'modality_subs', 'feature_subs', 'conclusion_subs', 'others_subs']
                    
                    for field in required_fields:
                        if field not in parsed_data:
                            raise ValueError(f"Missing required field: {field}")
                    
                    return {
                        'success': True,
                        'data': parsed_data,
                        'raw_response': content
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"{Fore.YELLOW}⚠️  JSON parsing failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        return {
                            'success': False,
                            'error': f"JSON parsing failed: {e}",
                            'raw_response': content
                        }
                        
            else:
                raise ValueError("No 'choices' field found in response")
                
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}❌ Request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            else:
                return {
                    'success': False,
                    'error': f"Request failed: {e}"
                }
                
        except Exception as e:
            print(f"{Fore.RED}❌ Unknown error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            else:
                return {
                    'success': False,
                    'error': f"Unknown error: {e}"
                }
    
    return {
        'success': False,
        'error': f"Reached maximum retry attempts ({max_retries})"
    }

def print_ai_response(content):
    """Print AI response in a formatted way"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}📊 AI Analysis Result")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.WHITE}{content}")

# Test function
if __name__ == "__main__":
    user_text_to_analyze = (
        "\"prediction\": \"I'll analyze this ophthalmologic image step by step to determine the appropriate next examinations.\\n\\n## Step 1: Image Analysis\\nLooking at this close-up photograph of an eye, I can observe:\\n- A cloudy, opaque lens (cataract)\\n- The pupil appears grayish-white instead of black\\n- The cornea appears clear\\n- This is consistent with a mature or dense cataract\\n\\n## Step 2: Clinical Context\\nWith a dense cataract present, several clinical considerations arise:\\n- Visual function assessment is needed\\n- The opacity prevents direct visualization of posterior structures\\n- Pre-operative evaluation may be necessary if surgery is considered\\n- Underlying pathology needs to be ruled out\\n\\n## Step 3: Evaluation of Each Option\\n\\n**A. Visual acuity test**\\n- Essential for documenting baseline visual function\\n- Helps determine the functional impact of the cataract\\n- Required for surgical planning\\n- **INDICATED**\\n\\n**B. Specular microscopy**\\n- Evaluates corneal endothelial cell count and morphology\\n- Important for cataract surgery planning\\n- Helps assess surgical risk\\n- **INDICATED**\\n\\n**C. Dilated fundus examination**\\n- The dense cataract prevents adequate visualization of the fundus\\n- Direct ophthalmoscopy would be impossible through this opacity\\n- **NOT FEASIBLE** with this degree of cataract\\n\\n**D. Ultrasound B-scan**\\n- Can visualize posterior segment structures when media is opaque\\n- Essential for detecting retinal detachment, vitreous hemorrhage, or other posterior pathology\\n- Critical before cataract surgery to ensure good visual potential\\n- **INDICATED**\\n\\n**E. CT scan**\\n- Not routinely indicated for cataract evaluation\\n- Would only be considered if trauma or orbital pathology suspected\\n- **NOT INDICATED** based on this presentation\\n\\n**F. Blood glucose measurement**\\n- Diabetes can cause cataracts and affect surgical outcomes\\n- Important for pre-operative assessment\\n- Helps identify systemic factors\\n- **INDICATED**\\n\\n## Final Answer\\n\\nThe appropriate examinations are: **A, B, D, F**\\n\\n- A. Visual acuity test (functional assessment)\\n- B. Specular microscopy (corneal endothelial evaluation)\\n- D. Ultrasound B-scan (posterior segment evaluation through opaque media)\\n- F. Blood glucose measurement (systemic evaluation)"
    )

    print(f"{Fore.YELLOW}🚀 Sending request to AI model...")
    result = analyze_text_with_retry(user_text_to_analyze)
    
    if result['success']:
        print(f"{Fore.GREEN}✅ Request successful!")
        print_ai_response(json.dumps(result['data'], indent=2, ensure_ascii=False))
    else:
        print(f"{Fore.RED}❌ Request failed: {result['error']}")
        if 'raw_response' in result:
            print(f"{Fore.YELLOW}Raw response: {result['raw_response']}")

