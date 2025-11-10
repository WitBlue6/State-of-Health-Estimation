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
        self.mode = None
        self.latest_image = None
        self.frame_rate = 20
        self.compression_quality = 95
        self._running = False
        self._display_thread = None
        self._loop = None
        self.frame = None  # 发送字符串帧
        self.list_data = None  # 发送[soh, thres]列表
        self.send = False
        self.logger = self.write_log
        self.log_path = "./outputs/web_log.txt"
        self.receive_buffer = bytearray()
        self.receive_info = {
            "type": None,
            "data": None
        }
        self.received = False
        self.on_receive_callback = None  # 接收回调函数

    async def _message_sender(self, websocket):
        """发送信息"""
        while self._running:
            try:
                if self.mode not in ['both', 'client']:
                    await asyncio.sleep(0.5)
                    continue
                if self.frame is not None and self.send:
                    self.send = False
                    buffer = self.frame
                    self.frame = None
                    msg = {
                        "type": "text",
                        "data": buffer,
                        "msg_id": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    }
                    # 发送
                    json_bytes = json.dumps(msg)
                    await websocket.send(json_bytes)
                    self.logger(f"已发送{msg['type']}:{msg['msg_id']}")
                        
                elif self.list_data is not None and self.send:
                    self.send = False
                    buffer = self.list_data
                    self.list_data = None
                    msg = {
                        "type": "list",
                        "data": buffer,
                        "msg_id": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    }
                    # 发送
                    json_bytes = json.dumps(msg)
                    await websocket.send(json_bytes)
                    self.logger(f"已发送{msg['type']}:{msg['msg_id']}")

                await asyncio.sleep(1 / self.frame_rate)

            except websockets.exceptions.ConnectionClosed as e:
                self.logger(f"发送时连接关闭, code={e.code}, reason={e.reason}")
                break
            except Exception as e:
                self.logger(f"发送错误: {e}")
                break

    async def _message_receiver(self, websocket):
        """接收信息"""
        self.receive_buffer = bytearray()
        while self._running:
            try:
                if self.mode not in ['both', 'server']:
                    await asyncio.sleep(0.5)
                    continue
                data = await websocket.recv()
                #self.logger("data:", data)
                # 正在接收数据
                if isinstance(data, str):
                    try:
                        # ######【注意】:接收后让处理线程将self.received清空#####
                        msg = json.loads(data)
                        if msg.get("type") == "text":
                            self.receive_info["data"] = msg["data"]
                            self.receive_info["type"] = msg["type"]
                            self.logger(f"收到消息:{msg.get('msg_id')}>>>{self.receive_info['data']}")
                            self.received = True
                            if self.on_receive_callback:
                                self.on_receive_callback(self.receive_info)

                        elif msg.get("type") == "list":
                            self.receive_info["data"] = msg["data"]
                            self.receive_info["type"] = msg["type"]
                            self.logger(f"收到消息:{msg.get('msg_id')}>>>{self.receive_info['data']}")
                            self.received = True
                            if self.on_receive_callback:
                                self.on_receive_callback(self.receive_info)
                        else:
                            self.logger(f"未知类型消息:{msg.get('type')}")
                    except Exception as e:
                        self.logger(f"接收时解码失败:{e}")
                        
            except websockets.exceptions.ConnectionClosed as e:
                self.logger(f"接收时连接关闭:code={e.code},reason={e.reason}")
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
        except asyncio.CancelledError:
            self.logger("任务被取消")
        except Exception as e:
            self.logger(f"处理器错误: {e}")
        finally:
            await websocket.close()
            self.logger("连接已关闭")

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
                    self.logger(f"连接成功!\n本机ip:{websocket.local_address} 对端ip:{websocket.remote_address}")
                    self._running = True
                    await self._handler(websocket)
                    return True
            except Exception as e:
                self.logger(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        asyncio.sleep(delay)
        self.logger("所有连接尝试均失败，请退出程序。")
        sys.exit(1)  # 超过N次连接失败，自动退出程序

    def run(self, mode='both'):
        import threading
        self.mode = mode
        if mode == 'server':
            threading.Thread(target=self.start_server, daemon=True).start()
            self._running = True
        elif mode == 'client':
            if not self.peer_uri:
                self.logger("❌ 客户端模式需要有效的 peer_uri, 当前为空")
                return
            def run_client():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._connect_with_retry())
            threading.Thread(target=run_client, daemon=True).start()
        elif mode == 'both':
            def run_both():
                self.start_server()
                if not self.peer_uri:
                    self.logger("❌ both模式需要有效的 peer_uri, 当前为空")
                    return
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._connect_with_retry())
            threading.Thread(target=run_both, daemon=True).start()

    def stop(self):
        """停止运行"""
        self._running = False
        if self._loop:
            self._loop.stop()
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join()

    def write_log(self, log: str, append=True):
        #print(log)
        if self.log_path:
            if append:
                with open(self.log_path, "a") as f:
                    f.write(log + "\n")
            else:
                with open(self.log_path, "w") as f:
                    f.write(log + "\n")