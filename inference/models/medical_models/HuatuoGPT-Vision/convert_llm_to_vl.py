#%%
import torch
import argparse
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Qwen2ForCausalLM,
)

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Convert Qwen2.5-LLM weights into Qwen2.5-VL")
    parser.add_argument("--vl_model_path", type=str, required=True, help="Path to the pretrained Qwen2.5-VL model")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to the pretrained Qwen2.5-LLM model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the converted Qwen2.5-VL model")
    return parser.parse_args()

#%%
def main():
    args = parse_args()

    # Load models
    print("Loading VL model from:", args.vl_model_path)
    model_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_model_path, torch_dtype="auto"
    )

    print("Loading LLM model from:", args.llm_model_path)
    model_lm = Qwen2ForCausalLM.from_pretrained(
        args.llm_model_path, torch_dtype="auto"
    )

    # Print model parameter info
    print("==== Parameters of model_vl ====")
    vl_params = list(model_vl.named_parameters())
    print(f"Total number of VL model parameters: {len(vl_params)}")
    print("Sample parameters:")
    for i, (name, _) in enumerate(vl_params[:10]):
        print(f"  {name}")

    print("\n==== Parameters of model_lm ====")
    lm_params = list(model_lm.named_parameters())
    print(f"Total number of LLM model parameters: {len(lm_params)}")
    print("Sample parameters:")
    for i, (name, _) in enumerate(lm_params[:10]):
        print(f"  {name}")

    # Parameter replacement
    print("\n==== Starting parameter replacement ====")
    lm_state_dict = model_lm.state_dict()
    vl_model_state_dict = model_vl.state_dict()
    new_vl_model_state_dict = {}

    replaced_count = 0
    skipped_count = 0

    for key in vl_model_state_dict.keys():
        if key in lm_state_dict and vl_model_state_dict[key].shape == lm_state_dict[key].shape:
            new_vl_model_state_dict[key] = lm_state_dict[key]
            replaced_count += 1
        else:
            new_vl_model_state_dict[key] = vl_model_state_dict[key]
            skipped_count += 1
            if key not in lm_state_dict:
                print(f"Parameter not found in LLM model: {key}")
            else:
                print(f"Shape mismatch - skipping: {key}, VL shape: {vl_model_state_dict[key].shape}, LLM shape: {lm_state_dict[key].shape}")

    model_vl.load_state_dict(new_vl_model_state_dict)
    print(f"\nParameter replacement completed! Replaced: {replaced_count}, Skipped: {skipped_count}")

    # Save model
    print(f"\nSaving converted VL model to: {args.save_path}")
    model_vl.save_pretrained(args.save_path)

    processor = AutoProcessor.from_pretrained(args.vl_model_path)
    processor.save_pretrained(args.save_path)
    print("Processor saved successfully.")

#%%
if __name__ == "__main__":
    main()
