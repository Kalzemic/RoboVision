import serial, struct, time, math

PORT = '/dev/ttyACM0'
BAUD = 115200
NUM_JOINTS = 2

def build_packet(joints):
    payload = struct.pack(f'<{NUM_JOINTS}f', *joints)
    return bytes([0xAA, 0x55]) + payload

ser = serial.Serial(PORT, BAUD)
t0 = time.time()
try:
    while True:
        t = time.time() - t0
        joints = [math.pi / 2 * math.sin(t + i * 0.5) for i in range(NUM_JOINTS)]
        ser.write(build_packet(joints))
        time.sleep(5e-6)
except KeyboardInterrupt:
    ser.close()