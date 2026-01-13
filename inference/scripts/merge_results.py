import json
import sys

def merge_results(original_file, new_file, output_file, replace_index):
    """
    Merge new results into the original results.
    
    Args:
        original_file: The original complete result file.
        new_file: The file containing a single new result.
        output_file: The output file after merging.
        replace_index: The index to replace.
    """
    # Read original results
    with open(original_file, 'r', encoding='utf-8') as f:
        original_results = json.load(f)
    
    # Read new results
    with open(new_file, 'r', encoding='utf-8') as f:
        new_results = json.load(f)
    
    if not new_results:
        print("New result file is empty")
        return
    
    # Get the item to replace
    new_item = new_results[0]
    
    # Replace or add
    found = False
    for i, item in enumerate(original_results):
        if item.get('index') == replace_index:
            original_results[i] = new_item
            found = True
            print(f"Replaced result for index {replace_index}")
            break
    
    if not found:
        original_results.append(new_item)
        original_results.sort(key=lambda x: int(x.get('index', 0)))
        print(f"Added result for index {replace_index}")
    
    # Save merged results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_results, f, indent=2, ensure_ascii=False)
    
    print(f"Merge completed, results saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge_results.py <original_file> <new_file> <output_file> <index>")
        sys.exit(1)
    
    merge_results(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))