import asyncio
import websockets
import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

clientes_conectados = set()

async def lidar_com_cliente(websocket):
    clientes_conectados.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clientes_conectados.remove(websocket)

async def reencaminhar_udp():
    loop = asyncio.get_event_loop()
    while True:
        try:
            
            data, addr = sock.recvfrom(2048)
            mensagem = data.decode('utf-8')
            
           
            if clientes_conectados:
                websockets.broadcast(clientes_conectados, mensagem)
        except BlockingIOError:
            pass 
        
        await asyncio.sleep(0.01)

async def main():
    print(f"A escutar YOLO no UDP: {UDP_IP}:{UDP_PORT}")
    print("A servir WebSockets na porta 8080...")
    
    
    async with websockets.serve(lidar_com_cliente, "127.0.0.1", 8080):
        await reencaminhar_udp()

if __name__ == "__main__":
    asyncio.run(main())