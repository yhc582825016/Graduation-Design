from datasets import load_dataset
data= load_dataset('/workspace/ye/dataset/yitingxie/rlhf-reward-datasets')
def func(data):
    data['prompt'] = '\n\nHuman:'+"<|endoftext|>".join(data['prompt'].split('\n\nHuman:')[1:])
    return data
data['train'].map(func)
