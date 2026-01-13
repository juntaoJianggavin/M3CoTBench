import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration
import os

model_path = "./pretrained/Qwen3-VL-30B-A3B-Instruct"  # Replace with your path

print("="*60)
print("Qwen3-VL Model Diagnostic")
print("="*60)

# 1. Check model files
print("\n[1] Checking model file integrity")
required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']
for file in required_files:
    path = os.path.join(model_path, file)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"  {file}: {'✓' if exists else '✗'} ({size} bytes)")

# 2. Load tokenizer
print("\n[2] Testing Tokenizer")
try:
    # Attempt loading slow tokenizer
    tokenizer_slow = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False
    )
    print(f"  Slow tokenizer: ✓ ({type(tokenizer_slow).__name__})")
    
    test_text = "Hello, this is a test. 你好，这是一个测试。"
    encoded = tokenizer_slow.encode(test_text, add_special_tokens=False)
    decoded = tokenizer_slow.decode(encoded, skip_special_tokens=True)
    
    print(f"  Original: {test_text}")
    print(f"  Encoded: {len(encoded)} tokens")
    print(f"  Decoded: {decoded}")
    print(f"  Match: {'✓' if test_text == decoded else '✗ Mismatch!'}")
    
    if test_text != decoded:
        print(f"\n  Discrepancy Analysis:")
        print(f"    Original Length: {len(test_text)}")
        print(f"    Decoded Length: {len(decoded)}")
        print(f"    Original Bytes: {test_text.encode('utf-8')[:50]}")
        print(f"    Decoded Bytes: {decoded.encode('utf-8')[:50]}")
        
except Exception as e:
    print(f"  Slow tokenizer: ✗ {e}")

try:
    # Attempt loading fast tokenizer
    tokenizer_fast = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    print(f"\n  Fast tokenizer: ✓ ({type(tokenizer_fast).__name__})")
    
    encoded = tokenizer_fast.encode(test_text, add_special_tokens=False)
    decoded = tokenizer_fast.decode(encoded, skip_special_tokens=True)
    
    print(f"  Decoded: {decoded}")
    print(f"  Match: {'✓' if test_text == decoded else '✗ Mismatch!'}")
    
except Exception as e:
    print(f"\n  Fast tokenizer: ✗ {e}")

# 3. Test special tokens
print("\n[3] Checking special tokens")
try:
    special_tokens = {
        'bos_token': tokenizer_slow.bos_token,
        'eos_token': tokenizer_slow.eos_token,
        'pad_token': tokenizer_slow.pad_token,
        'unk_token': tokenizer_slow.unk_token,
    }
    for name, token in special_tokens.items():
        print(f"  {name}: {token} (ID: {tokenizer_slow.convert_tokens_to_ids(token) if token else 'None'})")
except Exception as e:
    print(f"  Error: {e}")

# 4. Test vocab
print("\n[4] Checking vocabulary")
try:
    vocab_size = tokenizer_slow.vocab_size
    print(f"  Vocab size: {vocab_size}")
    
    # Test some common tokens
    test_tokens = ["the", "是", "hello", "世界"]
    for token in test_tokens:
        token_id = tokenizer_slow.convert_tokens_to_ids(token)
        decoded = tokenizer_slow.decode([token_id])
        print(f"  '{token}' -> ID {token_id} -> '{decoded}'")
except Exception as e:
    print(f"  Error: {e}")

# 5. Test processor
print("\n[5] Testing Processor")
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"  Processor: ✓ ({type(processor).__name__})")
    
    # Check processor's tokenizer
    if hasattr(processor, 'tokenizer'):
        print(f"  Processor.tokenizer: {type(processor.tokenizer).__name__}")
        test_decoded = processor.tokenizer.decode(encoded, skip_special_tokens=True)
        print(f"  Processor decode test: {test_decoded}")
        print(f"  Match: {'✓' if test_text == test_decoded else '✗'}")
except Exception as e:
    print(f"  Error: {e}")

# 6. Check model configuration
print("\n[6] Checking model configuration")
try:
    import json
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    print(f"  model_type: {config.get('model_type')}")
    print(f"  vocab_size: {config.get('vocab_size')}")
    print(f"  hidden_size: {config.get('hidden_size')}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*60)
print("Diagnostic complete")
print("="*60)