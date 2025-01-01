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

    
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from base import ABC, ChatModelBase, Path, Union, debug
import torch



# chat = [
#     {"role": "system", "content": "This is a conversation between an AI assistant(you) and a User. You, the AI assistant, should help to answer the questions from the User. "},
#     {"role": "system", "content": "Here is some history:"},
#     {"role": "user", "content": "Can you tell me where Tsinghua University is?"},
#     {"role": "assistant", "content": "Sure! Tsinghua University is at Beijing. "},
#     {"role": "system", "content": "Here is the new question:"},
#     {"role": "user", "content": content},
# ]
MODEL_SETTING = {
    k:v.strip() for k,v in {
    "base": "This is a conversation between an AI assistant and a User in a System. The AI \"assistant\" should answer the questions for the \"user\" according to the settings of \"system\".",
    "cat": "This is a conversation between a smart AI cat who speaks English(you) and a User. You, the AI assistant who is like a cat, should answer the questions from the User.",
}.items()}
USER = "user"
SYSTEM = "system"
ASSISTANT = "assistant"

class LocalChat(ChatModelBase, ABC):  
    __POST_PATTERN = re.compile(r'(.*[\.\?!。？！])(|\n|[^A-Za-z0-9\.\?!。？！])?')
    # __POST_PATTERN_LAST = re.compile(r'(*)')
    def __init__(
        self, model:str = 'base', 
        data_base_path:Union[str, Path]=None,
        model_path = './checkpoint-38820'
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
        self.__BASE_CHAT = MODEL_SETTING[model]
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__model = AutoModelForCausalLM.from_pretrained(model_path, device_map = self.__device)  
    
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
        # TODO: 结合知识库
        # prompt += 'Please use \"*hzw*\" as the end of reply.'
        answer_max_len = 2*len(prompt)
        # 1. 历史限制
        history = history or []
        while len(history)>0 and len(history)//2>=self.MAX_HISTORY_LEN:
            history = history.copy()
            history.pop(0)
            history.pop(0)
        # 2. 生成prompt
        chat_list = self.__add_history(history) + [
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
        # debug("reply: -----\n", reply), input('debugging out')
        # debug("-----")
        
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
        history = self.load_data_base() or []
        history += [
            {"role": USER, "content": "Can you tell me where SJTU is?"},
            {"role": ASSISTANT, "content": "SJTU(Shanghai JiaoTong University) is at Shanghai. This is a great university."},
        ]
        return "Sure! Tsinghua University is at Beijing. ", history
    
        