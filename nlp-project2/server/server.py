from pathlib import Path
WEB_DIR = Path(__file__).absolute().parent/'web'
import sys
sys.path.append(str(WEB_DIR.parents[1]))
from model import init_model, ChatModelBase
from http.server import HTTPServer, BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    USER_MODEL:dict[tuple[str, int], tuple[ChatModelBase, list]] = {}

    @staticmethod
    def __get_args(path:str)->tuple[dict[str], str]:
        args_begin = path.find('?')
        if args_begin==-1:
            return {}, path
        args_str = path[args_begin+1:]
        key_values = args_str.split('&')
        res = {}
        for s in key_values:
            k, v = s.split('=')
            if v.isdigit():
                v = int(v)
            elif(v.isnumeric()): 
                v = float(v)
            res[k]=v
        return res, path[:args_begin]

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=UTF-8')
        self.end_headers()
        args, pure_path = self.__get_args(self.path)
        model = args.get('model', None)
        database = args.get('db', None)
        if database is not None:
            database = 'E:\\python_works\\AILAB\\DB-GPT\\UltraDomain\\{}.jsonl'.format(database)
        character = args.get('character', 'base')
        # print(self.address_string())
        if self.address_string() not in self.USER_MODEL:
            # 加载不同模型
            model, _, history = init_model(model, database, character)
            self.USER_MODEL[self.address_string()] = [model, history]
            model.initialize()

        path = WEB_DIR/f'.{pure_path}'
        # print(path)
        if path.exists(): 
            with open(path, 'r', encoding='utf-8') as f:
                reply_str = f.read()
        else: reply_str = ''
        
        self.wfile.write(reply_str.encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=UTF-8')
        self.end_headers()
        reply = ''
        user_prompt = self.rfile.read(int(self.headers['content-length'])).decode()
        print(user_prompt)
        if user_prompt=='\\quit':
            if self.address_string() in self.USER_MODEL:
                self.USER_MODEL.pop(self.address_string())
                print(f'user {self.address_string()} quit, len: {len(self.USER_MODEL)}')
        else:
            model, history = self.USER_MODEL[self.address_string()]
            if user_prompt=='\\newsession':
                model.initialize()
            else:   
                reply, history = model.chat(user_prompt, history)
                self.USER_MODEL[self.address_string()][1] = history
                # print(datas)
        self.wfile.write(reply.encode())


if __name__=='__main__':
    # 开启服务器
    port = int(sys.argv[1]) if len(sys.argv)>1 else 10110
    addr_port = ('localhost', port)
    httpd = HTTPServer(addr_port, Handler)
    print(f'Please visit http://{addr_port[0]}:{addr_port[1]}/WebClientPage.html?model=lora')
    httpd.serve_forever()

