import socket
import json
import time

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
seq      = 0

CASAS   = {(0,0),(1,0),(0,1)}
ARVORES = {(7,0),(8,0),(9,0),(7,1),(8,1),(9,1),(7,2),(8,2)}
RIO     = {(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9)}
BLOQUEADAS = CASAS | ARVORES | RIO

def send(status, objs):
    global seq
    payload = {"seq": seq, "status": status, "objs": objs}
    sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
    seq += 1

def estado_base(personagem_pos, personagem_ori, inimigo_ori="N"):
    objs = []
    id_c = 10
    for pos in sorted(CASAS):
        objs.append({"id": id_c, "t": "casa",      "p": list(pos), "o": None})
        id_c += 1
    for pos in sorted(ARVORES):
        objs.append({"id": id_c, "t": "arvore",    "p": list(pos), "o": None})
        id_c += 1
    for pos in sorted(RIO):
        objs.append({"id": id_c, "t": "rio",       "p": list(pos), "o": None})
        id_c += 1
    objs.append({"id": 99, "t": "inimigo",     "p": [9,9], "o": inimigo_ori})
    objs.append({"id":  1, "t": "personagem",  "p": list(personagem_pos), "o": personagem_ori})
    return objs

def calcular_orientacao(pos_atual, pos_nova, ori_atual):
    c1, r1 = pos_atual
    c2, r2 = pos_nova
    if c2 > c1: return "E"
    if c2 < c1: return "W"
    if r2 > r1: return "S"
    if r2 < r1: return "N"
    return ori_atual

def step(pos, col, row, ori, inimigo_ori="N", delay=0.8):
    if (col, row) in BLOQUEADAS or not (0 <= col <= 9 and 0 <= row <= 9):
        return pos, ori
    nova_ori = calcular_orientacao(pos, (col, row), ori)
    nova_pos = (col, row)
    send("ok", estado_base(nova_pos, nova_ori, inimigo_ori))
    time.sleep(delay)
    return nova_pos, nova_ori

def andar(pos, ori, destinos, inimigo_ori="N", delay=0.8):
    for (c, r) in destinos:
        pos, ori = step(pos, c, r, ori, inimigo_ori, delay)
    return pos, ori

pos = (2, 2)
ori = "E"
send("reset", estado_base(pos, ori))
time.sleep(2)

# Desce para o sul junto das vilas
pos, ori = andar(pos, ori, [(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9)])
time.sleep(0.3)

# Anda para este até à passagem do rio (row 0 está livre)
pos, ori = andar(pos, ori, [(3,9),(4,9)])
time.sleep(0.3)

# Sobe até à passagem (5,0 está livre — rio começa em 5,1)
pos, ori = andar(pos, ori, [(4,8),(4,7),(4,6),(4,5),(4,4),(4,3),(4,2),(4,1),(4,0)])
time.sleep(0.3)

# Atravessa o rio pela passagem em (5,0)
pos, ori = andar(pos, ori, [(5,0)])
time.sleep(0.5)

# Entra no lado direito do rio
pos, ori = andar(pos, ori, [(6,0),(6,1),(6,2)])
time.sleep(0.3)

# Aproxima-se da floresta
pos, ori = andar(pos, ori, [(6,3),(6,4),(6,5),(6,6)])
time.sleep(0.5)

# Tenta entrar na floresta — bloqueado pelas árvores
pos, ori = andar(pos, ori, [(7,6),(7,5),(7,4),(7,3)])
time.sleep(0.5)

# Inimigo nota o personagem
send("ok", estado_base(pos, ori, inimigo_ori="W"))
time.sleep(1.5)

# Personagem sobe para ver a floresta de perto
pos, ori = andar(pos, ori, [(6,3),(6,2),(6,1),(6,0)], inimigo_ori="W")
time.sleep(0.3)

# Oclusão
send("occluded", [])
time.sleep(2)

send("ok", estado_base(pos, ori))
time.sleep(1)

# Regressa pela passagem do rio
pos, ori = andar(pos, ori, [(5,0),(4,0)])
time.sleep(0.3)

# Desce pelo lado esquerdo
pos, ori = andar(pos, ori, [(4,1),(4,2),(4,3),(3,3),(2,3),(2,2)])
time.sleep(0.3)

# Visita as vilas
pos, ori = andar(pos, ori, [(1,2),(1,3),(2,3),(2,2)])
time.sleep(1)

# Inimigo regressa ao normal
send("ok", estado_base(pos, ori, inimigo_ori="N"))
time.sleep(2)

sock.close()
