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
        peer_uri="ws://localhost:3016/ws/receive"
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
soh = 100
rul = 350
threshold = 75
while True:
    cnt += 1
    if cnt == 16:
        send_message(transfer, "日志摘要：设备健康度持续下降，预示着可能存在故障模块且模块出现频次较高。同时，健康度长期逼近阈值，表明设备健康状况面临严重威胁。此外，RUL_关键模块失效和RUL_依赖模块失效分别显示了关键和依赖模块可能出现故障的次数及剩余寿命范围。而RUL_互补模块失效则提供了相关模块失效的信息。\n\n建议执行操作：\n\n1. **检查电源模块**：查看电源模块的健康度，确保其在健康度下，如低于阈值但高于0，确认模块没有明显故障或性能问题。如果发现任何异常情况，立即进行排查和修复。\n\n2. **评估系统状态**：通过监控系统运行状态，包括设备整体运行率、关键模块运行率、依赖模块运行率以及RUL等指标，了解设备的整体运行情况。若发现某个关键或依赖模块运行率显著降低，需要进一步调查原因并采取相应的措施，例如更换或优化相关模块。\n\n3. **更新RUL预测**：基于系统当前状态和历史数据，对关键和依赖模块的RUL进行更新，确保RUL始终处于可接受范围内。同时，考虑未来可能发生的故障情况，制定详细的故障恢复计划，以便在RUL值过低时能够快速启动备用方案。\n\n4. **定期维护和升级**：针对设备老化、硬件损坏等问题，定期进行设备维护和升级，保证设备的稳定运行和关键模块的正常工作。这不仅有助于提高设备的整体健康度，还可以预防潜在的故障发生。\n\n5. **强化安全防护**：对于可能存在的恶意攻击或软件故障，应加强网络安全防护措施，包括安装最新的防病毒软件、防火墙等，防止恶意软件对设备造成影响。\n\n6. **定期备份数据**：定期备份重要数据，以防设备因故障导致的数据丢失。这样即使设备出现问题，也可以从备份中恢复数据，避免业务中断。\n\n7. **定期进行健康体检**：定期对设备进行全面健康检查，包括硬件、软件、网络等各个层面，及时发现并解决潜在的问题，保持设备的健康状态。\n\n通过上述操作，可以有效提高设备的健康度，减少设备故障风险，延长设备使用寿命，保障系统的稳定运行。")
        cnt = 0
        continue
    soh -= random.random()
    threshold = soh * 0.8 + (random.random() - 0.5) * 2
    warning = 0
    rul -= 2 * random.random()
    log = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "系统正常"
    soh_buffer = [float(soh), float(threshold), warning, float(rul), log]
    send_soh(transfer, soh_buffer)
    time.sleep(3)
    print("Send")