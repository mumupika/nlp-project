# 聊天机器人使用文档

本项目为NLP-project-2的实现.

## 仓库结构
```shell
.
├── README.md
├── main.py
├── model
│   ├── __init__.py
│   ├── api_model.py
│   ├── base.py
│   ├── character_settings.yaml
│   └── local_model.py
├── model_test
│   ├── test outcome, ends with .txt
│   └── test programs, ends with .py
└── server
    ├── server.py
    └── web
        └── ...
```
        
## 环境配置

```shell
pip install -r requirements.txt
```

该模型的使用方式有两种: 命令行使用与网页使用. 

## 命令行

```shell
python main.py [--model <your_model>] [--db <your_databsase>] [--character <your_character>]
```

知识库、人物设定和模型选择都要通过命令行传参来决定: 
- `model`参数: 可选, 传入模型的模式. 可选`base`/`lora`/`api`三个. 分别是微调的0.5b模型, 微调的1.5b模型, 和利用api进行交流的模型. 默认`base`
- `character`参数: 可选, 传入模型的人物性格. 目前通过prompt engineering实现`base`, `base_cn`, `cat_cn`三种, 默认`base`. 在`model/character_settings.yaml`文件中可以设置更多的性格和设定. 
- `db`参数: 可选, 传入知识库文件路径, 目前只支持**jsonl**和**txt**格式文件. **特别注意:** `api`模式没有实现结合知识库的功能. 

输入"\quit"退出, 输入"\newsession"开启新对话, 其余为正常聊天.

## 网页
粗略实现了一个**不是很安全**的服务器, 由于路径问题请*谨慎*使用知识库. 
```shell
python server/server.py [<your_port>]
```

- `port`参数: 可选, 传入希望服务器开启的端口号, 默认是10110. 运行之后可以根据输出的提示访问相关网页. 

model, character, db参数通过网址GET传参来传递, 具体见[`命令行`](#命令行)里写的文档. 
传参示例: http://localhost:10110?model=lora&character=cat_cn

## 评测相关

在model_test目录下实现了大模型的相关内容评测和输出结果。

```shell
cd model_test
python <any_test_program> [ > <your_log_file>]
```