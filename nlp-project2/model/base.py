from abc import ABC, abstractmethod
from typing import Union, Literal
from pathlib import Path
import json
class Colored:
    __COLORED_CODE = {
        'default': '\033[0m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }
    def __init__(self, color:Literal['red', 'green', 'blue', 'default', 'yellow'] = 'default'): self.set(color)
    def __enter__(self): return self
    def __exit__(self, *args, **kwargs): self.set('default')
    def set(self, color:Literal['red', 'green', 'blue', 'default', 'yellow'] = 'default'): print(self.__COLORED_CODE[color], end='')
        
def debug(*args, color = 'green', **kwargs):
    # with Colored(color) as color_ctrl: print(*args, **kwargs), color_ctrl.set('red'), input('\n@hzw: input something to continue...')
    pass
    
class ChatModelBase(ABC):
    def __init__(self, data_base_path:Union[str, Path] = None):
        ''' Chat Model Base Constructor
        
        Params:
        ---
            data_base_path (str|Path): The path to the database file
        '''
        super().__init__()
        self.__data_base_path = data_base_path
        self.MAX_HISTORY_LEN = 30
    
    @abstractmethod
    def chat(self, prompt:str, history:list[str]=None)->tuple[str, list[str]]:
        ''' Chat Function

        Params:
        ---
            prompt (str): The given prompt
            history (list[str]): The conversation history
        
        Returns:
        ---
            out (tuple[str, list[str]]): the `reply` and `history`
        '''
        raise NotImplementedError
    
    def load_data_base(self):
        #TODO: 加载数据集怎么做得更好些，应当用另一个history来帮助(如果让用API的话), 否则计划直接塞进prompt/history里。
        # debug(reply)
        if self.__data_base_path is None: return
        history:list[dict[str, str]] = []
        with open(self.__data_base_path, 'r', encoding='utf-8') as f:
            while f.readable():
                l = f.readline()
                if l=='': break
                obj:dict[str] = json.loads(l)
                history += [
                    {'role':'user', 'content': obj['input']},
                    {'role':'assistant', 'content': obj['answers'][0]},
                ]
                # debug(q_a)
        self.MAX_HISTORY_LEN = len(history)//2+10
        return history
        
                 
    @abstractmethod
    def initialize(self)->tuple[str, list[str]]:
        raise NotImplementedError
    
    # INIT_PROMPTS = {
    #     'api': '我将要和你进行对话，也许需要你结合一些我给出的知识。听懂了就可以了。',
    #     # 'base': '我将要和你进行对话，也许需要你结合一些我给出的知识。听懂了回复“是”', 
    #     'base': "Expert, I have some questions to ask you. Can you give me a hand?",
    #     'neko': '你是一只可爱的猫娘，我将和你进行一些对话，请你用猫娘的口吻回复我哟~',
    #     'others': '你是一个性格为{}的小助手，我要和你进行对话。听懂了就可以了。'
    #     #TODO: 更多性格的调教
    # }
    # EXTRACT_PROMPT_TEMPLATE = (
    # "A question is provided below. Given the question, extract up to "
    # "keywords from the text. Focus on extracting the keywords that we can use "
    # "to best lookup answers to the question.\n"
    # "Generate as more as possible synonyms or alias of the keywords "
    # "considering possible cases of capitalization, pluralization, "
    # "common expressions, etc.\n"
    # "Avoid stopwords.\n"
    # "Provide the keywords and synonyms in comma-separated format."
    # "Formatted keywords and synonyms text should be separated by a semicolon.\n"
    # "---------------------\n"
    # "Example:\n"
    # "Text: Alice is Bob's mother.\n"
    # "Keywords:\nAlice,mother,Bob;mummy\n"
    # "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    # "Keywords:\nPhilz,coffee shop,Berkeley,1982;coffee bar,coffee house\n"
    # "---------------------\n"
    # "Text: {text}\n"
    # "Keywords:\n")
