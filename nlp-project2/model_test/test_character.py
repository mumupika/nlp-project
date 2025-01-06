from tqdm import tqdm
from pathlib import Path
MODEL_DIR = Path(__file__).absolute().parents[1]/'model'
import sys
sys.path.append(str(MODEL_DIR.parent))

from model import init_model


model, _, history = init_model("lora", character="cat_cn")

prob_list = [
    """请介绍一下你作为猫猫的主要职责和日常工作。""",
    """如果你生活在2000年，你认为当时的社会环境对你的职业有什么影响？""",
    """如果你的朋友突然取消了和你的约会，你会怎么反应？""",
    """你认为人工智能在未来会如何影响你的职业？""",
    """如果你的同事在工作中犯了一个错误，你会怎么处理？""",
    """请描述你在不同场合下的着装风格。""",
    """如果你有机会改变自己的一个性格特点，你会选择改变什么？为什么？"""
]

for idx, prob in tqdm(enumerate(prob_list)):
    print(f"Q{idx+1}: {prob} \n")
    reply, history = model.chat(prob, history)
    print(f"A: {reply} \n")