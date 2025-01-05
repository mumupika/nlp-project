from abc import ABC, abstractmethod
from typing import Union, Literal
from pathlib import Path
import os
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class Colored:
    __COLORED_CODE = {
        'default': '\033[0m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }
    def __init__(self, color:Literal['red', 'green', 'blue', 'default', 'yellow'] = 'default'): 
        """set the color for later `print`

        Params:
        ---
            color (Literal[&#39;red&#39;, &#39;green&#39;, &#39;blue&#39;, &#39;default&#39;, &#39;yellow&#39;], optional): the color you choose. Defaults to 'default'.
        
        Examples:
        ---
            >>> with Colored('red'):
            ...     print(1)
            1
            >>> color = Colored('yellow')
            >>> print(2)
            >>> color.set('red')
            >>> print(3)
        """
        self.set(color)
    def __enter__(self): return self
    def __exit__(self, *args, **kwargs): self.set('default')
    def set(self, color:Literal['red', 'green', 'blue', 'default', 'yellow'] = 'default'): print(self.__COLORED_CODE[color], end='')
        
def debug(*args, color = 'green', **kwargs):
    """debug out. delete "#" to use."""
    # with Colored(color) as color_ctrl: print(*args, **kwargs), color_ctrl.set('red'), input('\n@hzw: input something to continue...')
    pass
    
class ChatModelBase(ABC):
    def __init__(self, data_base_path:Union[str, Path] = None):
        """Chat Model Base Constructor
        
        Params:
        ---
            data_base_path (Union[str, Path], optional): The path to the database file. Defaults to None.
        """
        super().__init__()
        self.database=None
        self.__load_data_base(data_base_path)
        self.MAX_HISTORY_LEN = 30
    
    @abstractmethod
    def chat(self, prompt:str, history:list[str]=None)->tuple[str, list[str]]:
        ''' Chat Function

        Params:
        ---
            prompt (str): The given prompt
            history (list[str]): The conversation history. Defaults to None.
        
        Returns:
        ---
            out (tuple[str, list[str]]): the `reply` and `history`
        '''
        raise NotImplementedError
    
    def __load_data_base(self, data_base_path:Path = None):
        # 路径检查
        if data_base_path is None: return
        if not isinstance(data_base_path, Path):
            data_base_path = Path(data_base_path)
        data_base_path = data_base_path.absolute()
        if not data_base_path.exists():
            raise FileNotFoundError(f'File {data_base_path} not exist')
        
        if data_base_path.name[-6:]=='.jsonl':
            # JSONLoader加载json数据
            loader = JSONLoader(
                data_base_path, 
                jq_schema='{question: .input, answer: .answers[0], meta: .meta}', 
                json_lines=True,
                text_content=False
            )
        elif data_base_path.name[-4:]=='.txt':
            # TextLoader加载普通文字
            loader = TextLoader(data_base_path, encoding='utf-8')
        else:
            raise RuntimeError("Cannot accept other documents")
        
        # 读取、切割
        splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=10)    
        documents = splitter.split_documents(loader.load())
        # 向量化、存变量
        embeddings = HuggingFaceEmbeddings(
            model_name = "moka-ai/m3e-base",
            model_kwargs = {'device':'cpu'},
            encode_kwargs = {'normalize_embeddings': True},
            # query_instruction = '为json生成向量表示用于文本检索'
        )
        self.database = Chroma.from_documents(documents, embeddings)
        
                 
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
