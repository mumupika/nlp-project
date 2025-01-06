from tqdm import tqdm
from pathlib import Path
MODEL_DIR = Path(__file__).absolute().parents[1]/'model'
import sys
sys.path.append(str(MODEL_DIR.parent))

from model import init_model
from tqdm import tqdm

lora_model, _, lora_history = init_model("lora")
db_model, _, db_history = init_model("lora", "./database/刘邦.txt")

prob_list = [
    """高祖的出身和家庭背景是怎样的？""",
    """高祖的外貌特征有哪些？""",
    """高祖的性格和才能有哪些？""",
    """高祖在成为皇帝之前有哪些重要经历？""",
    """高祖与吕公和吕后的关系如何？""",
    """高祖在起义过程中有哪些重要战役和事件？""",
    """高祖在建立汉朝后有哪些重要举措？""",
    """高祖在位期间有哪些重要事件？""",
    """高祖的去世和葬礼是怎样的？""",
]


for idx, prob in tqdm(enumerate(prob_list)):
    print(f"Q{idx+1}: {prob} \n")
    lora_reply, lora_history = lora_model.chat(prob, lora_history)
    print(f"A: {lora_reply} \n")
    db_reply, db_history = db_model.chat(prob, db_history)
    print(f"AD {db_reply} \n")
    _, lora_history = lora_model.initialize()
    _, db_history = db_model.initialize()