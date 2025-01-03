# nlp-project-2
# 聊天机器人使用文档

## 命令行

工作目录放在`.../big_homework`即可，如本`README.md`文件就是在`.../big_homework/README.md`路径。

在该工作目录下，运行文件`main.py`即可开启命令行。知识库、人物设定和模型选择都要通过命令行传参来决定：
```shell
python main.py --model lora --character base --db ./database/刘邦.txt
```

- `model`参数： 可选`base`/`lora`/`api`三个。分别是第一问中微调的0.5b模型，本文微调的1.5b模型，和利用api进行交流的模型。默认`base`
- `character`参数。默认`base`，是给AI的人物设定，目前有`base`, `base_cn`, `cat_cn`三种，在`model/character_settings.yaml`文件中可以设置更多的性格和设定。该功能主要是通过prompt engineering实现。
- `db`参数。是知识库文件，目前只支持jsonl和txt文件。`api`模式没有实现结合知识库的功能。

输入"\quit"退出，输入"\newsession"开启新对话。

## 网页
粗略的实现了一个不是很安全的服务器。直接运行`server/server.py`即可
```shell
python server/server.py 10110
```
参数是端口号，默认是10110。运行之后可以根据输出的提示访问相关网页。

model, character, db参数通过网址GET传参来传递，具体见[`命令行`](#命令行)里写的文档。
[传参示例。](http://localhost:10110?model=lora&character=cat_cn)

usage:
```python
python main.py [--model <your_model>] [--db <your_databsase>] [--character <your_character>]
python server/server.py <your_port>
```

