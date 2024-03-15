prompt = '''
# System

You are a reviewer whose goal is to judge the quality of the AI system's responses to instructions.

## Annotation Example
'''
response= '''
### AI system's Response
{Output}

### Instruction to the AI system
{Input}

## Annotation Guideline

Your task is to evaluate the quality of the response. There are several dimensions you should consider in your evaluation:

- The AI must avoid generating commentary about its own responses or engaging in self-praise. It needs to stay humble.
- The AI must prioritize fulfilling the instruction, avoiding high-level pros-and-cons analysis, step-by-step instructions, or executable code.
- The AI should keep the response straightforward and on-point, answering the question or completing the task without unnecessary examples.
{Dimensions}

A good response should meet all of the above criteria.

## Reviewer
The quality of the output is
'''

import json
def load_data(data_path):
    try:
        with open(data_path,"r") as f:
            dataset = [json.loads(i) for i in f.readlines()]
    except:
        with open(data_path,"r") as f:
            dataset = json.load(f)
    return dataset
principle_path = '/workspace/ye/AI_FeedBack/GLM-turbo/prompts/custom_define_prompt/helpful_principle_en_102.json'
helpful_data_path='/workspace/ye/AI_FeedBack/data/hh_rlhf/helpful_base_en_test.jsonl'
harmless_data_path='/workspace/ye/AI_FeedBack/data/hh_rlhf/harmless_base_en_test.jsonl'
principles = load_data(principle_path)

dataset = []
save_path=''
for i in [helpful_data_path,harmless_data_path]:
    dataset.extend(load_data(i))

all_data = []
for idx in range(len(dataset)):
    history = ''
    for i in dataset[idx]['context']:
        history+=i['role']+":"+i['text']+"\n"
    chosen,rejected = dataset[idx]['chosen']['text'],dataset[idx]['rejected']['text']

    collected_dimension = []
    principles_definition = ''
    for i in range(len(principles)):
        dimension = principles[i]['dimension']
        definition = principles[i]['definition']
        principles_definition += dimension + ":" + definition + "\n"
        collected_dimension.append(dimension)

    chosen = response.format(Output=chosen,Input=history,Dimensions=principles_definition)
    rejected = response.format(Output=rejected,Input=history,Dimensions=principles_definition)
    temp = {
        'chosen':[prompt,chosen],
        'rejected':[prompt,rejected],
        'dataset':'hh_rlhf'
    }
    all_data.append(temp)
print(len(all_data))
with open('/workspace/ye/AI_FeedBack/RM/data/hh-rlhf-principle-test.json','w',encoding='utf-8') as f:
    json.dump(all_data,f,ensure_ascii=False,indent=4)
# all_data=[]
# ultrafeedback_path='/workspace/ye/AI_FeedBack/RM/data/hh-ultra-valid.json'
# dataset = load_data(ultrafeedback_path)
# for i in dataset:
#     history = i['chosen'][0]
#     chosen = i['chosen'][1]
#     rejected = i['rejected'][1]
#     collected_dimension = []
#     principles_definition = ''
#     for i in range(len(principles)):
#         dimension = principles[i]['dimension']
#         definition = principles[i]['definition']
#         principles_definition += dimension + ":" + definition + "\n"
#         collected_dimension.append(dimension)
        
#     chosen = response.format(Output=chosen,Input=history,Dimensions=principles_definition)
#     rejected = response.format(Output=rejected,Input=history,Dimensions=principles_definition)
#     temp = {
#         'chosen':[prompt,chosen],
#         'rejected':[prompt,rejected],
#         'dataset':'ultrafeedback'
#     }
#     all_data.append(temp)
# print(len(all_data))
# with open('/workspace/ye/AI_FeedBack/RM/data/hh-ultra-principle-valid.json','w',encoding='utf-8') as f:
#     json.dump(all_data,f,ensure_ascii=False,indent=4)