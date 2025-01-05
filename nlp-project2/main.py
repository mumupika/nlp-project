from argparse import ArgumentParser
from model import init_model, Colored

WAITING = 'Waiting...'

def mainloop():
    # 解析命令行参数
    parser = ArgumentParser()
    parser.add_argument('--character', type=str, default='base')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--db', type=str, default=None) #E:\python_works\AILAB\DB-GPT\UltraDomain\cs.jsonl
    args = parser.parse_args()
    
    # 主循环
    model, _, history = init_model(args.model, args.db, args.character)
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
        elif (prompt == '') or (prompt=='\n'):
            continue
        print(WAITING, end=' ', flush=True)
        reply, history = model.chat(prompt, history)
        color.set('green')
        print("\b"*(len(WAITING)+1)+'A: ', end='')
        color.set('default')
        print(reply)

if __name__=='__main__':
    mainloop()
