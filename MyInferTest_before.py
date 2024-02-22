from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
    
model_dir = snapshot_download('qwen/Qwen-1_8B-Chat-Int4')
    
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True
).eval()

with open('city.txt','r',encoding='utf-8') as fp:
    city_list=fp.readlines()
    city_list=[line.strip().split(' ')[1] for line in city_list]

Q='青岛4月6日下雨么?'

prompt_template='''
给定一句话：“%s”，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON样式，格式为{"city":城市,"date":日期}

请问，这个语句中城市、日期信息为：
'''
prompt=prompt_template%(Q,)
#A,hist=model.chat(tokenizer,Q,history=None)
for _ in range(10):
    A,hist=model.chat(tokenizer,prompt,history=None)
    print('Q:%s\nA:%s\n'%(Q,A))
print("The End")
