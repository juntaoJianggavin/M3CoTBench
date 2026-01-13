#%%
import re
import json,os
from collections import defaultdict
import random
import ast
import wandb
# import argparse
import traceback

import difflib

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity >= highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index

def match_choice3(text,options):
    matches = list(re.finditer(r"(is |是|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )([abcdefghijklmnABCDEFGHIJKLMN])(\W|$)", text, re.S))
    if matches:
        ans = matches[0].group(2)
        return ans,1

    text = text.lower()

    opsindex = [(opt,text.rindex(options[opt].lower())) for opt in options if options[opt].lower() in text]
    if len(opsindex) > 0:
        return sorted(opsindex,key=lambda x:x[1],reverse=True)[0][0],2
    
    oplabels = [x for x in options]
    opans = [options[x].lower() for x in options]
    ansindex = find_most_similar_index(opans,text.lower())
    return oplabels[ansindex], 3

def test_OmniMedVQA(da):
    a,b,c,d = da.get('option_A'), da.get('option_B'), da.get('option_C'), da.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    # print(f'answer_list {answer_list}')
    # print(f"answer_list {answer_list[find_most_similar_index(answer_list, da['model_output'])] == da['gt_answer']}")
    return answer_list[find_most_similar_index(answer_list, da['model_output'])] == da['gt_answer']

def test_OmniMedVQA_new(da):
    answer_list = da['options']
    choice = find_most_similar_index(answer_list, da['model_output'])
    label = answer_list[choice] == da['answer']
    return label,choice

def test_choice_simple(da):
    choice = ['A','B','C','D','E','F','G'].index(da['model_output'][0])    
    answer_list = da['options']
    label = answer_list[choice] == da['answer']
    return label,choice


wrong_ans = []
def test_mmlu(da):
    global wrong_ans
    ans = match_choice3(da['model_output'],da)
    # ans = match_choice3(da['huatuo_answer'],da)
    if len(ans) == 0:
        wrong_ans.append(da)
        ans = 'A'
    return ans[0] == da['answer']


wrong_data = []
sample_num = 5
wrong_ans_num = 0
def test_choice_llava(input_data):
    global wrong_ans_num
    type2score = {}
    sub_type2score = {}
    sample_data = {}
    ty_set = set()
    res = {}

    miss_match_num = 0

    if len(input_data) == 0:
        return sample_data,res
    
    for da in input_data:
        ty = da['dataset']
        
        if ty not in type2score:
            type2score[ty] = [0,0]

        type2score[ty][1] += 1
        
        try:
            opt_title = ['A','B','C','D','E','F','G','H','I','J','K']
            if da['answer'] == '?':
                da['is_correct'] = False
            else:
                answer_label = opt_title[da['options'].index(da['answer'])]
                da['answer'] = answer_label
                da['is_correct'] = test_mmlu(da)
        except Exception as e:
            print(da)
            traceback.print_exc()
            assert False,da

        if da['is_correct']:
            type2score[ty][0] += 1
        else:
            wrong_data.append(da)
        
        def update_sub_record(sub_type2score,type_name,iscorrect):
            if type_name not in sub_type2score:
                sub_type2score[type_name] = [0,0]
            sub_type2score[type_name][1] += 1
            if iscorrect:
                sub_type2score[type_name][0] += 1
        if 'subset' in da and da['subset']!= '':
            update_sub_record(sub_type2score, f'{ty}_{da["subset"]}',da['is_correct'])


    for k,v in type2score.items():
        print(f'【{k}】Accuracy：{(v[0]/v[1] if v[0] > 0 else 0) :.4f}   question number:{v[1]}')
        res[k] = (v[0]/v[1] if v[0] > 0 else 0)

    print(f'The total score for multiple-choice questions：{sum([sc[0] for k,sc in type2score.items() if "___" not in k ])/len(input_data) :.3f}   question number: {len(input_data)}')
    res['The total score for multiple-choice questions'] = sum([sc[0] for k,sc in type2score.items() if "___" not in k ])/len(input_data)

    print('\n'+f'-------'*4)
    for k,v in sub_type2score.items():
        print(f'【{k}】Accuracy：{(v[0]/v[1] if v[0] > 0 else 0) :.4f}   question number:{v[1]}')
        res[k] = (v[0]/v[1] if v[0] > 0 else 0)

    sample_data = [{'Input':x['query'],'Output':x['model_output'],'Answer': x['answer'], 'Dataset':x['dataset']} for x in random.sample(input_data,5)]
    print(f'\n wrong_ans_num {wrong_ans_num}')
    return sample_data,res

def test_chat(chat_data,sample_data):
    for da in chat_data:
        ty = da['dataset']
        da['answer'] = da['output']
        if sample_data is not None:
            sample_data[ty] = []
            if len(sample_data[ty]) < sample_num:
                sample_data[ty].append(da)
    return sample_data


def score_mix_llava(datas, iswandb = False):
    global test_func
    test_func = match_choice3
    filted_data = []
    test_id_set = set()
    for da in datas:
        if 'question_type' in da and da['question_type'] == 'open':
            continue
        if da['test_id'] in test_id_set:
            continue
        if len(da['options']) == 0:
            continue
        test_id_set.add(da['test_id'])
        filted_data.append(da)
    print(f'filted_data: {len(filted_data)} datas:{len(datas)}')

    sample_data,res = test_choice_llava(filted_data)    

    if iswandb:
        table = wandb.Table(columns=["Input", "Output","Answer","Dataset"])
        for da in sample_data:
            table.add_data(da['Input'],da['Output'],da['Answer'],da['Dataset'])
        res['InputOutputTable'] = table
    return res

