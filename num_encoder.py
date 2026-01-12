import numpy as np

def pos_encode(ints, min_length=3):
    n = len(ints)
    
    result = {
        "pos": [],        # 记录中间的长0位置
        "stripped": []    # 记录非0数值和中间的短0
    }
    
    i = 0
    while i < n:
        if ints[i] == 0:
            # 开始探测 0 的长度
            start_idx = i
            while i < n and ints[i] == 0:
                i += 1
            
            # --- 优化点：检测是否到达末尾 ---
            if i == n:
                # 如果这些 0 一直延伸到数组结束
                # 无论长短，都直接丢弃（隐含在 length 中）
                break 
            
            # 如果不是末尾的 0，按原逻辑处理
            run_length = i - start_idx
            if run_length >= min_length:
                result["pos"].append([start_idx, run_length])
            else:
                result["stripped"].extend([0] * run_length)
        else:
            result["stripped"].append(ints[i])
            i += 1
            
    return result

def pos_decode(pos, stripped, original_length):
    result = []
    stripped_idx = 0
    
    # 1. 还原中间部分（逻辑不变）
    for start_idx, length in pos:
        needed = start_idx - len(result)
        if needed > 0:
            result.extend(stripped[stripped_idx : stripped_idx + needed])
            stripped_idx += needed
            
        result.extend([0] * length)
    
    # 2. 补上 stripped 中剩余的数据
    if stripped_idx < len(stripped):
        result.extend(stripped[stripped_idx:])
        
    # 3. --- 优化点：利用原始长度还原末尾隐含的零 ---
    current_len = len(result)
    if current_len < original_length:
        result.extend([0] * (original_length - current_len))
        
    return result

def encode_pos_arr(pos) -> bytes:
    out = bytearray()
    for start_idx, length in pos:
        out.extend(prefix_encode(start_idx))
        out.extend(prefix_encode(length))
    return bytes(out)

def decode_pos_arr(f) -> list[list[int]]:
    pos = []
    while True:
        try:
            start_idx = prefix_decode(f)
            length = prefix_decode(f)
            pos.append([start_idx, length])
        except EOFError:
            break
    return pos


def prefix_encode(n: int) -> bytes:
    if n < 0:
        raise ValueError("仅支持非负整数")
    
    # 1字节: 0xxxxxxx (0 ~ 127)
    if n <= 0x7F:
        return bytes([n])
    
    # 2字节: 10xxxxxx xxxxxxxx (0 ~ 16,383)
    # 前缀 10 (2 bits), 数据位 14 bits
    elif n <= 0x3FFF:
        return bytes([0x80 | (n >> 8), n & 0xFF])
    
    # 3字节: 110xxxxx xxxxxxxx xxxxxxxx (0 ~ 2,097,151)
    # 前缀 110 (3 bits), 数据位 21 bits
    elif n <= 0x1FFFFF:
        return bytes([0xC0 | (n >> 16), (n >> 8) & 0xFF, n & 0xFF])
    
    # 4字节: 111xxxxx xxxxxxxx xxxxxxxx xxxxxxxx (0 ~ 536,870,911)
    # 前缀 111 (3 bits), 数据位 29 bits
    elif n <= 0x1FFFFFFF:
        return bytes([0xE0 | (n >> 24), (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])
    
    else:
        raise ValueError("数值超过4字节编码上限 (最大 536,870,911)")

import io

def prefix_decode(f: io.IOBase) -> int:
    # 读取首字节
    first_byte = f.read(1)
    if not first_byte:
        raise EOFError("Reach end of stream")
    
    first = first_byte[0]
    
    # 1字节: 0xxxxxxx
    if (first & 0x80) == 0:
        return first
    
    # 2字节: 10xxxxxx xxxxxxxx
    elif (first & 0xC0) == 0x80:
        remaining = f.read(1)
        if len(remaining) < 1: raise ValueError("Truncated 2-byte sequence")
        return ((first & 0x3F) << 8) | remaining[0]
    
    # 3字节: 110xxxxx xxxxxxxx xxxxxxxx
    elif (first & 0xE0) == 0xC0:
        remaining = f.read(2)
        if len(remaining) < 2: raise ValueError("Truncated 3-byte sequence")
        return ((first & 0x1F) << 16) | (remaining[0] << 8) | remaining[1]
    
    # 4字节: 111xxxxx xxxxxxxx xxxxxxxx xxxxxxxx
    elif (first & 0xE0) == 0xE0:
        remaining = f.read(3)
        if len(remaining) < 3: raise ValueError("Truncated 4-byte sequence")
        return ((first & 0x1F) << 24) | (remaining[0] << 16) | (remaining[1] << 8) | remaining[2]
    
    else:
        raise ValueError(f"Invalid prefix: {bin(first)}")

def encode_scale(scale):
    """将 float32 scale 转换为 8-bit uint8"""
    if scale <= 1e-14: return 0 # 极小值归零
    # 取 log2 并映射到 0-255
    # 假设我们关心的范围是 -40 到 0 (2^-40 约等于 1e-14)
    log_s = np.log2(scale)
    clipped_log = np.clip(log_s, -40, 0) 
    quantized = np.round((clipped_log + 40) * (255 / 40))
    return int(quantized)

def decode_scale(q_scale):
    """将 8-bit uint8 还原为 float32 scale"""
    if q_scale == 0: return 0
    log_s = (q_scale * (40 / 255)) - 40
    return 2 ** log_s

# --- 测试 ---
if __name__ == "__main__":
    arr = [4,0,2,3,0,0,0,0,0,2,7,5,0,4,0,3,0,0,0,0,0,5,7,3,6,4,0,0,0,0,4,4,6,3]
    pos = pos_encode(arr)
    print(pos)
    print(encode_pos_arr(pos['pos']))
    print(decode_pos_arr(io.BytesIO(encode_pos_arr(pos['pos']))))
