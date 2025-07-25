import asyncio
import sys
import websockets
import numpy as np
import json
from datetime import datetime

CHUNK_SIZE = 1024 * 1024


class RealTimeTransfer():
    def __init__(self, my_ip, my_port, peer_uri):
        self.my_ip = my_ip
        self.my_port = my_port
        self.peer_uri = peer_uri
        self.latest_image = None
        self.frame_rate = 20
        self.compression_quality = 95
        self._running = False
        self._display_thread = None
        self._loop = None
        self.frame = None
        self.send = False
        self.logger = print
        self.receive_buffer = bytearray()
        self.receive_info = None
        self.received = False

    async def _message_sender(self, websocket):
        """发送信息"""
        while self._running:
            try:
                if self.frame is not None and self.send:
                    self.send = False
                    buffer = self.frame
                    msg = {
                        "type": "text",
                        "data": buffer,
                        "msg_id": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    }
                    json_bytes = json.dumps(msg).encode("utf-8")
                    await websocket.send(json_bytes)
                    self.logger(f"已发送:{msg['msg_id']}")
                await asyncio.sleep(1 / self.frame_rate)

            except websockets.exceptions.ConnectionClosed:
                self.logger("发送时连接关闭")
                break
            except Exception as e:
                self.logger(f"发送错误: {e}")
                break

    async def _message_receiver(self, websocket):
        """接收信息"""
        self.receive_buffer = bytearray()
        while self._running:
            try:
                self.received = False
                data = await websocket.recv()
                # 正在接收数据
                if isinstance(data, bytes):
                    try:
                        msg = json.loads(data.decode("utf-8"))
                        if msg.get("type") == "text":
                            self.receive_info = msg["data"]
                            self.logger(f"收到消息:{msg.get('msg_id')}>>>{self.receive_info}")
                            self.received = True
                        else:
                            self.logger(f"未知类型消息:{msg.get('type')}")
                    except Exception as e:
                        self.logger(f"接收时解码失败:{e}")
                        
            except websockets.exceptions.ConnectionClosedOK:
                self.logger("连接正常关闭")
                break
            except Exception as e:
                self.logger(f"接收错误: {e}")
                break

    async def _handler(self, websocket):
        """处理单个连接会话"""
        try:
            await asyncio.gather(
                self._message_sender(websocket),
                self._message_receiver(websocket)
            )
        except Exception as e:
            self.logger(f"处理器错误: {e}")
        finally:
            await websocket.close()

    def start_server(self):
        """启动WebSocket服务器"""

        async def _run_server():
            async with websockets.serve(
                    self._handler,
                    self.my_ip,
                    self.my_port,
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=1,
                    max_size=10 * 1024 * 1024,
            ):
                self.logger(f"服务器已启动 ws://{self.my_ip}:{self.my_port}")
                self._running = True
                await asyncio.Future()  # 永久运行

        def run_in_thread():
            asyncio.run(_run_server())

        import threading
        threading.Thread(target=run_in_thread, daemon=True).start()

    async def _connect_with_retry(self, max_retries=25, delay=3):
        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                        self.peer_uri,
                        ping_interval=30,
                        ping_timeout=60,
                        close_timeout=1
                ) as websocket:
                    self.logger("连接成功!")
                    await self._handler(websocket)
                    return True
            except Exception as e:
                self.logger(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        asyncio.sleep(delay)
        self.logger("所有连接尝试均失败，请退出程序。")
        sys.exit(1)  # 超过N次连接失败，自动退出程序

    def _start_server_and_connect(self):
        """在子线程运行的服务器和连接逻辑"""
        self.start_server()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_with_retry()) # 尝试连接ip

    def run(self):
        # 在子线程启动服务器
        import threading
        threading.Thread(target=self._start_server_and_connect, daemon=True).start()

    def stop(self):
        """停止运行"""
        self._running = False
        if self._loop:
            self._loop.stop()
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join()