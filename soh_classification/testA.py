from websocket_transfer import RealTimeTransfer
import time
import asyncio
import websockets

def send_message(transfer: RealTimeTransfer, message: str):
    transfer.frame = message
    transfer.send = True

async def test_websocket_connection(uri):
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ 成功连接到 {uri}")
            await websocket.send("ping")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                print(f"📩 收到响应: {response}")
            except asyncio.TimeoutError:
                print("⚠️ 连接成功但没有响应（可能对方不回消息）")
    except Exception as e:
        print(f"❌ 无法连接到 {uri}：{e}")

if __name__ == "__main__":

    transfer = RealTimeTransfer(
        my_ip="192.168.10.206",
        my_port=8765,
        peer_uri="ws://192.168.10.199:8765"
    )
    transfer.run()
    cnt = 1
    while True:
        if transfer._running:
            send_message(transfer, message=f"Hello LZH {cnt}")
            cnt += 1
            time.sleep(1)
