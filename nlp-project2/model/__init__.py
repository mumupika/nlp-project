import sys
from pathlib import Path
THIS_DIR = Path(__file__).absolute().parent
sys.path.append(str(THIS_DIR))

from api_model import APIChat
from local_model import LocalChat
from base import ChatModelBase, Colored

sys.path.pop()
PROJ_DIR = Path(__file__).absolute().parents[1]
CHECKPOINTS = {
    'base': PROJ_DIR/'checkpoint-38820',
    'lora': (PROJ_DIR/'checkpoint-17254', PROJ_DIR/'Qwen2.5-1.5B')
}
# print(LOCAL_MODEL_CHECKPOINT), exit(0)

def init_model(model:str = None, database:str=None, character:str = 'base')->tuple[ChatModelBase, str, list[str]]:
    '''
    初始化模型
    
    Params:
    ---
        model (str): 模型类型，是基础版还是增强版，默认是none（基础版）
        database (str): 知识库，默认是none
        character (str): 模型性格。默认是base。
    '''
    if model=='api':
        res_model = APIChat(data_base_path=database)
        reply, history = res_model.initialize()
        return res_model, reply, history

    model = model if model in CHECKPOINTS else 'base'
    res_model = LocalChat(character = character, model_path=CHECKPOINTS[model], data_base_path=database)
    
    reply, history = res_model.initialize()
    return res_model, reply, history
    
