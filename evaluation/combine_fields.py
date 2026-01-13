import pandas as pd
import json

input_file = "../inference/dataset/M3CoTBench.xlsx"
output_file = 'output_with_key_annotation_steps_final.xlsx' 


df = pd.read_excel(input_file)
def combine_fields(row):
    fields = [
        row.get('modality', ''),
        row.get('feature', ''),
        row.get('key_conclusion', ''),
        row.get('additional_analysis', ''),
    ]

    result = {
        "solution1": fields[0] if fields[0] and not pd.isna(fields[0]) else None,
        "solution2": fields[1] if fields[1] and not pd.isna(fields[1]) else None,
        "solution3": fields[2] if fields[2] and not pd.isna(fields[2]) else None,
        "solution4": fields[3] if fields[3] and not pd.isna(fields[3]) else None,
    }
    return json.dumps(result, ensure_ascii=False, indent=4)

df['key_annotation_steps'] = df.apply(combine_fields, axis=1)


df.to_excel(output_file, index=False)

print(f'save as {output_file}')