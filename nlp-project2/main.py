from argparse import ArgumentParser
from model import init_model, Colored

WAITING = 'Waiting...'

def mainloop():
    # 解析命令行参数
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--db', type=str, default=None) #E:\python_works\AILAB\DB-GPT\UltraDomain\cs.jsonl
    args = parser.parse_args()
    # 加载模型、Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained() #TODO: 加载tokenizer，可能不在这写
    #model, reply, history = init_model(args.model, args.database) #TODO: 在这个函数里，我们根据参数先行“调教”
    model, _, history = init_model(args.model, args.db)\
    # 主循环
    color = Colored('yellow')
    while True:
        color.set('yellow')
        print('Q: ', end='')
        color.set('default')
        prompt = input()
        if prompt == '\\quit':
            break
        elif prompt == '\\newsession':
            reply, history = model.initialize()
            continue
        print(WAITING, end=' ', flush=True)
        reply, history = model.chat(prompt, history)
        color.set('green')
        print("\b"*(len(WAITING)+1)+'A: ', end='')
        color.set('default')
        print(reply)

if __name__=='__main__':
    mainloop()