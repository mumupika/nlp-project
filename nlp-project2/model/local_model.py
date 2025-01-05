from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from base import ABC, ChatModelBase, Path, Union, debug
import torch, yaml
from peft import PeftModel

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
    __POST_PATTERN = re.compile(r'(.*[,\.\?!，。？！])(|\n|[^A-Za-z0-9\.\?!。？！])?', re.M)
    # __POST_PATTERN_LAST = re.compile(r'(*)')
    def __init__(
        self, model_path:Union[str, tuple[str,str]] = './checkpoint-38820', 
        data_base_path:Union[str, Path]=None,
        character: str = 'base'
    ):
        ''' Chat Model Base Constructor
        
        Params:
        ---
            model_path(str | tuple[str,str]): The path to the model.
            data_base_path (str|Path): path of your database file. Defaults to None.
            character (str): The type of the model.
        '''
        super().__init__(data_base_path=data_base_path)
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__BASE_CHAT = CHARACTER_SETTING[character]['init_prompt']
        self.__INIT_HISTORY = CHARACTER_SETTING[character]['init_history']  

        if isinstance(model_path, tuple):
            base_model_path = str(model_path[1]) #base
            model_path = str(model_path[0]) #lora
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map = self.__device)
            self.__model = PeftModel.from_pretrained(base_model, model_path, device_map = self.__device)
        else:
            self.__model = AutoModelForCausalLM.from_pretrained(model_path, device_map = self.__device)
            
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)
            
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
        answer_max_len = 2*len(prompt)
        # 1. 历史限制
        history = history or []
        while len(history)//2>=self.MAX_HISTORY_LEN:
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