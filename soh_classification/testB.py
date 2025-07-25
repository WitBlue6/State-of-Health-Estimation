from websocket_transfer import RealTimeTransfer


if __name__ == "__main__":
    receicer = RealTimeTransfer(
        my_ip="192.168.10.199",
        my_port=8765,
        peer_uri="ws://192.168.10.206:8765",
    )
    receicer.run()
    print("OK")
    while True:
        pass