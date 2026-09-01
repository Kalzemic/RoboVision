import serial, struct 


NUM_JOINTS = 2

PORT = '/dev/ttyACM0'

BAUD = 115200


ser = serial.Serial(PORT,BAUD)

try:
    while True:
        
        angles = []
        print(f'Input {NUM_JOINTS} DOF:')
        for j in range(NUM_JOINTS):
            print(f'DOF {j}:')
            angle = float(input())
            angles.append(angle)

        payload  = bytes([0xAA,0x55]) + struct.pack(f'<{NUM_JOINTS}f',*angles) 

        ser.write(payload)
        print('Payload sent')
except KeyboardInterrupt:
    ser.close()