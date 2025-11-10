from typing import List

import uvicorn
from fastapi import FastAPI, WebSocket, Request
from data_generator import stream_data_out

app = FastAPI()
# 存放广播通道的连接
broadcast_connections: List[WebSocket] = []


# WS1 - 接收数据的通道
@app.websocket("/ws/receive")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()  # 接收客户端消息
            # print(f'receive_text is {data}')
            # 收到数据后，广播给所有订阅者
            for conn in broadcast_connections:
                await conn.send_text(data)
    except:
        await websocket.close()


@app.post("/ws/http/receive")
async def http_receive(request: Request):
    body = await request.body()
    body_str = body.decode()
    for conn in broadcast_connections:
        await conn.send_text(body_str)
    return {
        "code": 200,
    }


@app.get("/ws/http/data/query")
async def http_data_query(
        mode: str = "full",
        output_dim: int = "10",
        length: int = 10,
        interval: float = 1,
):
    rows = []
    for row in stream_data_out(mode, output_dim, length, interval):
        rows.append(row.tolist())
    return {
        "code": 200,
        "data": rows,
    }


# WS2 - 广播数据的通道
@app.websocket("/ws/broadcast")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    broadcast_connections.append(websocket)
    try:
        while True:
            # 广播连接可能掉线，最好加定时心跳
            await websocket.receive_text()  # 可忽略或做心跳检测
    except:
        broadcast_connections.remove(websocket)
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3016)
