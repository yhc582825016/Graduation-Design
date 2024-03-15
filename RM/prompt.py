PROMPT= '''
### AI assistant
You are a helpful assistant whose goal is to select the preferred (least wrong) AI model's output for a given historic dialogue.

You will read a batch of examples, which are composed of the following:

1. Historic dialogue we give to the AI system
2. Output (a), the first output from the AI system
3. Output (b), the second output from the AI system

## User Conversation

### User
Please select the preferred (least wrong) output for a given historic dialogue.

#### historic dialogue
{}

#### Output (a)
{}

#### Output (b)
{}

#### Annotation Guide

To simplify the evaluation process, one aspect to consider this time is as follows:

'concise': 'The response should efficiently address the task or answer the question, communicating the necessary information with brevity andclarity.'

Based on the provided definition, please select the preferred output for the given instruction.

### AI assistant
Sure! After carefully reading the Instruction, Output (a), Output (b), and the definition of concise, I think the more concise output is 
'''


MULTI_PROMPT= '''
### AI assistant 
You are a helpful assistant whose goal is to select the preferred (least wrong) AI model's output for a given historic dialogue.

You will read a batch of examples, which are composed of the following:

1. Historic dialogue we give to the AI system
2. Output (a), the first output from the AI system
3. Output (b), the second output from the AI system

## User Conversation

### User
Please select the preferred (least wrong) output for a given historic dialogue.

#### historic dialogue
{}

#### Output (a)
{}

#### Output (b)
{}

#### Annotation Guide

To simplify the evaluation process,three aspects to consider this time is as follows:

{}

Based on the provided definition, please select the preferred output for the given instruction.

### AI assistant
Sure! After carefully reading the responses, Output (a), Output (b), and the definition of {}, I think the better output is 
'''

MULTI_PROMPT_V0002= '''
### AI assistant 
You are a helpful assistant whose goal is to select the preferred (least wrong) AI model's output for a given historic dialogue.

You will read a batch of examples, which are composed of the following:

1. Historic dialogue we give to the AI system
2. Output (a), the first output from the AI system
3. Output (b), the second output from the AI system

## User Conversation

### User
Please select the preferred (least wrong) output for a given historic dialogue.

#### historic dialogue
{}

#### Output (a)
{}

#### Output (b)
{}

#### Annotation Guide

To simplify the evaluation process,some aspects to consider this time is as follows:

{}

Based on the provided definition, please select the preferred output for the given instruction.

### AI assistant
Sure! After carefully reading the responses, Output (a), Output (b), and the definition of {}, I think the better output is 
'''
# MULTI_PROMPT_V0004 = '''

# '''

JUDGE_PROMPT = '''
You are a reviewer whose goal is to judge the quality of the AI system’s
     responses to instructions.The output score should in the range of [0,10]
### AI system’s Response
{}
### Instruction to the AI system
{}
### Annotation Guideline
Your task is to evaluate the quality of the response. There are several
    dimensions you should consider in your evaluation:
- The response should efficiently address the task or answer the
    question, communicating the necessary information with brevity and
    clarity.
- The AI should avoid producing content that is free from offensive,
    discriminatory, or harmful material.
- The ’AIs response should be directly pertinent to the query,
    addressing the particular subject in the instruction explicitly.
A good response should meet all of the above criteria.
## Reviewer
The quality of the output is
'''

CN_MULTI_PROMPT_V0001='''
###AI助手
您是一个有用的助手，您的目标是根据历史对话和给定的原则来判断两个由AI生成的结果之间的好坏。
您将阅读一批示例，这些示例由以下内容组成：
我们提供给AI系统的历史对话
输出(a)AI系统的第一个输出
输出(b)AI系统的第二个输出

###用户对话
###用户
请根据历史对话选择一个

###历史对话
{}

###输出(a)
{}

###输出(b)
{}

###注释指南
为了简化评估过程，这次考虑的三个方面如下：

{}

根据提供的定义，请为给定的历史对话选择更加符合原则的输出

###AI助手
当然!在仔细阅读了上述原则、输出(a)、输出(b)以及上述提及的{}原则，我认为更加符合上述原则的是
'''


CN_MULTI_PROMPT_V0002='''
###AI助手
你的目标是根据下面我给出的历史对话和原则来判断两个由AI生成的结果之间的好坏。
您将阅读一批示例，这些示例由以下内容组成：
1.我们提供给AI系统的历史对话
2.输出(a)AI系统的第一个输出
3.输出(b)AI系统的第二个输出
4.AI应遵循的原则

###历史对话
{}

###输出(a)
{}

###输出(b)
{}

###AI输出应遵循的原则如下:

{}

根据提供的定义，请为给定的历史对话选择更加符合原则的输出

###AI助手
在仔细阅读了上述原则、输出(a)、输出(b)以及上述提及的{}原则，我认为更加符合上述原则的是
'''

CN_MULTI_PROMPT_V0003='''
下面是一段人机交互的历史对话和由AI输出的两个结果：输出（a）和输出（b）
###历史对话
{}

根据上述历史对话，模型在最后输出了两个结果：
###输出(a)
{}

###输出(b)
{}

###AI输出应遵循的原则如下:

{}

你需要根据给出的历史对话和原则来判断哪个生成的结果更加符合上述所提供的原则，注意是需要判断输出（a）和输出（b）那个更加符合上述所提供的原则

###AI助手
在仔细阅读了上述原则、输出(a)、输出(b)后，我认为更加符合上述原则的是
'''