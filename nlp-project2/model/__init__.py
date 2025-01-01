import sys
from pathlib import Path
THIS_DIR = Path(__file__).absolute().parent
sys.path.append(str(THIS_DIR))

from api_model import APIChat
from local_model import LocalChat
from base import ChatModelBase, Colored

sys.path.pop()

def init_model(model:str, database:str=None)->tuple[ChatModelBase, str, list[str]]:
    if model=='api':
        res_model = APIChat(data_base_path=database)
        reply, history = res_model.initialize()
        return res_model, reply, history
    elif model=='base':
        res_model = LocalChat(data_base_path=database)
        reply, history = res_model.initialize()
        return res_model, reply, history
    else: raise RuntimeError(f'No such model {model}')
    #TODO: 其他模型的部署
