import requests, json
from base import ABC, ChatModelBase, Path, Union

class APIChat(ChatModelBase, ABC):
    def __init__(
        self, model_url:str = "https://api.siliconflow.cn/v1/chat/completions", 
        data_base_path:Union[str, Path]=None
    ):
        super().__init__(data_base_path=data_base_path)
        self.__url = model_url
        self.__headers = {
            "Authorization": "Bearer sk-dxooypfhkpuqjyumyolguzchqpswxwrjgkjfwpvxksebvtjq",
            "Content-Type": "application/json"
        }
    
    def chat(self, prompt:str, history:list[str]=None)->tuple[str, list[str]]:
        # TODO: 结合知识库
        messages = [{
            "role": "user",
            "content": prompt
        }]
        if (history is not None) and len(history)>0:
            messages.append({
                "role": "system",
                "content": "Here is some history: "+'\n'.join(history)
            })
            if len(history)>=self.MAX_HISTORY_LEN:
                history.pop(1)
        payload = {
            "model": "Qwen/Qwen2-7B-Instruct",
            "messages": messages
        }
        response = requests.request("POST", self.__url, json=payload, headers=self.__headers)
        response = json.loads(response.text)
        reply = response['choices'][0]['message']['content']
        if history is None:
            history = []
        history.append(f'Q: {prompt}; A: {reply}')
        return reply, history
    
    def initialize(self):
        # self.load_data_base()
        return self.chat("我将要和你进行对话，也许需要你结合一些我给出的知识。听懂了就可以了。")