

def crc16(bytes):
    crc = 0xFFFF
    
    for b in bytes:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc = crc >> 1
    
    return crc

import io

START_STOP = 0x7E
ESCAPE = 0x7D

def encapsulate_frame(frame_bytes: bytes) -> bytes:
    out = bytearray()
    out.append(START_STOP) # 起始符

    for b in frame_bytes:
        if b == ESCAPE or b == START_STOP:
            out.append(ESCAPE)
            out.append(b ^ 0x20)
        else:
            out.append(b)
    return bytes(out)

def decapsulate_frame(f: io.IOBase) -> bytes:
    """
    流式读取：寻找起始 7E，读取数据直到遇见下一个 7E。
    """
    # 步骤 1: 必须先找到起始符 7E
    # 我们需要消耗掉所有非 7E 的数据（可能是噪音）
    while True:
        lead = f.read(1)
        if not lead: return None # 流结束
        if lead[0] == START_STOP:
            break

    out = bytearray()

    # 步骤 2: 读取内容
    while True:
        # 使用 peek 检查下一个字节而不消耗它 (如果 io 支持)
        # 或者直接 read，如果发现是 7E，则需要通过 seek 退回一个字节
        b = f.read(1)
        if not b: 
            return bytes(out) # 流意外结束，返回已读到的部分
        
        curr = b[0]
        
        if curr == START_STOP:
            # 关键：我们遇到了下一帧的开头。
            # 我们需要把这个 7E “还”给流，以便下一次调用能读到它
            if f.seekable():
                f.seek(f.tell() - 1)
            return bytes(out)
            
        if curr == ESCAPE:
            next_b = f.read(1)
            if next_b:
                out.append(next_b[0] ^ 0x20)
            else:
                break # 格式错误
        else:
            out.append(curr)

# --- 测试 ---
if __name__ == "__main__":
    raw_data = b"\x01\x7E\x03\x7D\x05"
    enc = encapsulate_frame(raw_data)
    print(f"Encapsulated: {enc.hex(' ')}") 
    # 应输出: 7e 01 7d 5e 03 7d 5d 05 7e
    
    dec = decapsulate_frame(enc)
    print(f"Decapsulated matches: {dec == raw_data}")