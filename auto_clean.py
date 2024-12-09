import os,shutil,time
num = 500
while True:
    print("Try cleaning...")
    if os.path.exists(f'./output/checkpoint-{num}') and os.path.exists(f'./output/checkpoint-{num+500}'): 
        shutil.rmtree(f'./output/checkpoint-{num}')
        num+=500
    time.sleep(60)