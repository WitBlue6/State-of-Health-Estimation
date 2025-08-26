from websocket_transfer import RealTimeTransfer
import time
import random
from datetime import datetime
def send_message(transfer: RealTimeTransfer, message: str):
    transfer.frame = message
    transfer.send = True
    while transfer.send:  # 等待发送完成
        time.sleep(0.01)

def send_soh(transfer: RealTimeTransfer, message: list):
    transfer.list_data = message
    transfer.send = True    
    while transfer.send:  # 等待发送完成
        time.sleep(0.01)

transfer = RealTimeTransfer(
        my_ip="172.17.1.97",
        my_port=8765,
        peer_uri="ws://117.133.23.34:5585/ws/receive"
    )
transfer.log_path = "./outputs/log_ws_debug.txt"
transfer.write_log("正在启动新的client(标识ID:LZH-DEBUG)", append=False)
transfer.run(mode="client")
send_cnt = 128
    
# 清空日志内容
    
while transfer._running == False:
    time.sleep(1)
    print("等待连接...")
# 循环读取数据
print("连接成功, 模型开始运行...")
cnt = 0
soh = 80
threshold = 75
while True:
    cnt += 1
    if cnt == 16:
        send_message(transfer, "[DEBUG]>>我是大模型输出，当前时间:" + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        cnt = 0
        continue
    soh += (random.random() - 0.5) * 2
    threshold = soh * 0.8 + (random.random() - 0.5) * 2
    warning = random.randint(0, 3)
    rul = random.randint(0, 350)
    log = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + ">>>LZH-DEBUG"
    soh_buffer = [float(soh), float(threshold), warning, float(rul), log]
    send_soh(transfer, soh_buffer)
    time.sleep(3)