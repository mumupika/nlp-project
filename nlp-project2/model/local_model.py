# from transformers import AutoTokenizer, AutoModelForCausalLM
# import re
# from base import ABC, ChatModelBase, Path, Union

# HISTORY_ITEM_TEMPLATE = '''
# User: {prompt}
# Expert: {reply}
# '''
# PROMPT_TEMPLATE = '''This is a conversation between a User and an Expert. 
# The Expert will only answer the questions, he won't ask the User to do something.
# Here begins their conversation:
# User: Hi, Expert!
# Expert: Hi, User!
# {history}
# User: {prompt}
# '''
# WITH_DATABASE_PROMPT_TEMPLATE = '''There is an Expert that just finished reading this article below, and he learnt something from it:
# {data_base}
# A User, who has some questions, met the Expert and began a conversation after the Expert finished learning:

# User: Hi, Expert!
# Expert: Hi, User!
# {history}
# User: {prompt}
# '''
# class LocalChat(ChatModelBase, ABC):  
#     __POST_PATTERN = re.compile(r'(.*[\.\?!。？！\n\"])')
#     def __init__(
#         self, model_path:Union[Path, str] = './checkpoint-38820', 
#         data_base_path:Union[str, Path]=None
#     ):
#         ''' Chat Model Base Constructor
        
#         Params:
#         ---
#             model_path (str|Path): The path to the model file
#             data_base_path (str|Path): The path to the database file
#         '''
#         super().__init__(data_base_path=data_base_path)
#         self.__prompt_template = PROMPT_TEMPLATE if data_base_path is None else WITH_DATABASE_PROMPT_TEMPLATE
#         self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.__model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'cuda')  
    
    
#     def chat(self, prompt:str, history:list[str]=None)->tuple[str, list[str]]:
#         ''' Chat Function

#         Params:
#         ---
#             prompt (str): The given prompt
#             history (list[str]): The conversation history
        
#         Returns:
#         ---
#             out (tuple[str, list[str]]): the `reply` and `history`
#         '''
#         # TODO: 结合知识库
#         # 1. 记录历史(1)
#         answer_max_len = len(prompt)*2+1
#         history_item = HISTORY_ITEM_TEMPLATE.format(prompt = prompt, reply = '{reply}')
#         # 2. 结合历史
#         if (history is not None) and (len(history)>0):
#             history = history.copy()
#             if len(history)>=self.MAX_HISTORY_LEN:
#                 history.pop(1)
#             # 生成prompt
#             prompt = self.__prompt_template.format(history = '\n'.join(history).strip(), prompt = prompt)
#         else:
#             history = []
#             prompt = self.__prompt_template.format(history = '', prompt = prompt)
#         # 3. 喂给模型    
#         inputs = self.__tokenizer([prompt], return_tensors="pt").to('cuda')
#         generated_ids = self.__model.generate(
#             **inputs,
#             pad_token_id = self.__tokenizer.eos_token_id,
#             max_new_tokens = answer_max_len,
#             top_p=0.95,
#             temperature=0.9,
#             do_sample=True
#         )
#         reply = self.__tokenizer.decode(
#             generated_ids[0],
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )[len(prompt):]
        
#         # 4. 答案处理
#         start_mark = reply.find('Expert:')
#         end_mark = reply.find('User:')
#         start_mark = 0 if start_mark==-1 else 7
#         end_mark = None if end_mark==-1 else end_mark
#         # debug(reply+f"\n-{start_mark}--{end_mark}-")
#         reply = reply[start_mark:end_mark]
#         # input(prompt)
#         r = re.search(self.__POST_PATTERN, reply)
#         reply = r.group() if r is not None else 'OK.'
#         reply = reply.strip()
        
#         # -1. 记录历史(2)
#         history.append(history_item.format(reply = reply).strip())
        
#         return reply, history
    
#     def initialize(self):
#         data_base = self.load_data_base()
#         self.__prompt_template = self.__prompt_template.format(history = '{history}', prompt = '{prompt}', data_base = data_base)
#         return self.chat( "Expert, I have some questions to ask you. Can you give me a hand?")

    
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
from base import ABC, ChatModelBase, Path, Union, debug
import torch, yaml
from peft import PeftModel



# chat = [
#     {"role": "system", "content": "This is a conversation between an AI assistant(you) and a User. You, the AI assistant, should help to answer the questions from the User. "},
#     {"role": "system", "content": "Here is some history:"},
#     {"role": "user", "content": "Can you tell me where Tsinghua University is?"},
#     {"role": "assistant", "content": "Sure! Tsinghua University is at Beijing. "},
#     {"role": "system", "content": "Here is the new question:"},
#     {"role": "user", "content": content},
# ]
with open(
    Path(__file__).absolute().parent/'character_settings.yaml', 
    encoding='utf-8'
) as chs:
    CHARACTER_SETTING = yaml.safe_load(chs)
    # print(CHARACTER_SETTING)


USER = "user"
SYSTEM = "system"
ASSISTANT = "assistant"

class LocalChat(ChatModelBase, ABC):  
    __POST_PATTERN = re.compile(r'(.*[,\.\?!，。？！])(|\n|[^A-Za-z0-9\.\?!。？！])?')
    # __POST_PATTERN_LAST = re.compile(r'(*)')
    def __init__(
        self, model_path:Union[str, tuple[str,str]] = './checkpoint-38820', 
        data_base_path:Union[str, Path]=None,
        character = 'base'
    ):
        ''' Chat Model Base Constructor
        
        Params:
        ---
            model (str|Path): The type of the model, for example, `base`, `cat`
            data_base_path (str|Path): The path to the database file
            model_path(str): The path to the model.
        '''
        super().__init__(data_base_path=data_base_path)
        # self.__BASE_CHAT = [
        #     {"role": "system", "content": MODEL_SETTING[model]},
        # ]
        isLora = isinstance(model_path, tuple)
        if isLora:
            base_model_path = str(model_path[1]) #base
            model_path = str(model_path[0]) #lora
        self.__BASE_CHAT = CHARACTER_SETTING[character]['init_prompt']
        self.__INIT_HISTORY = CHARACTER_SETTING[character]['init_history']
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not isLora:
            self.__model = AutoModelForCausalLM.from_pretrained(model_path, device_map = self.__device)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map = self.__device)
            self.__model = PeftModel.from_pretrained(base_model, model_path, device_map = self.__device)
            
        # self.__streamer = TextStreamer(self.__tokenizer)
    
    def __add_history(self, history:list[dict[str,str]] = None)->list[dict[str,str]]:
        if history is None or len(history)==0:
            return []
        return [{'role': SYSTEM, 'content': 'Here is some history'}] + history
        
    def chat(self, prompt:str, history:list[dict[str,str]] = None)->tuple[str, list[dict[str,str]]]:
        ''' Chat Function

        Params:
        ---
            prompt (str): The given prompt
            history (list[str]): The conversation history
        
        Returns:
        ---
            out (tuple[str, list[str]]): the `reply` and `history`
        '''
        # prompt += 'Please use \"*hzw*\" as the end of reply.'
        answer_max_len = 2*len(prompt)
        # 1. 历史限制
        history = history or []
        while len(history)>0 and len(history)//2>=self.MAX_HISTORY_LEN:
            history = history.copy()
            history.pop(0)
            history.pop(0)
        # 2. 生成prompt
        chat_list = []
        if self.database is not None:
            knowledges = self.database.similarity_search(prompt)
            debug(knowledges, color='blue')
            klg_str=''
            for item in knowledges:
                c = item.model_dump()['page_content']
                try:
                    item = eval(c)
                    klg_str += f'question: {item["question"]}\nanswer: {item["answer"]} According to {item["meta"]}\n'
                except:
                    klg_str += c
            chat_list = [{'role': SYSTEM, 'content': 'Here list some knowledges that the assistant knew: \n'+klg_str}]
        
        chat_list += self.__add_history(history) + [
            {'role': SYSTEM, 'content': 'Here comes new conversations:'},
            {'role': USER, 'content': prompt}
        ]
        chat_inputs = self.__BASE_CHAT + self.__tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True)
        # 3. 喂给模型    
        inputs = self.__tokenizer([chat_inputs], return_tensors="pt", add_special_tokens=True).to(self.__device)
        generated_ids = self.__model.generate(
            **inputs,
            pad_token_id = self.__tokenizer.eos_token_id,
            max_new_tokens = answer_max_len,
            top_p=0.9,
            temperature=0.9,
            do_sample=True
        )
        reply = self.__tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )[0]
        L = len(chat_list)
        
        debug('chat_inputs--------\n', chat_inputs, color='yellow')

        # 4. 答案处理
        reply = reply[len(chat_inputs)-len('<|im_end|>')*L - len('<|im_start|>')*(L+1):]
        
        debug("origin reply-----------------\n", reply)
        r = re.search(self.__POST_PATTERN, reply)
        # if r is None:
        #     r = re.search(self.__POST_PATTERN_LAST, reply)
        reply = r.group() if r is not None else 'OK.'
        reply = reply.strip()
        
        # -1. 记录历史(2)
        history = history + [
            {'role': USER, 'content': prompt},
            {'role': ASSISTANT, 'content': reply}
        ]
        return reply, history
    
    def initialize(self):
        reply = ''
        history = []
        for obj in self.__INIT_HISTORY:
            history += [
                {'role': role, 'content': content}
                for role, content in obj.items()
            ]
            reply = obj['assistant']
        return reply, history
    
        
