import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from typing import Tuple, Optional
import numpy as np
from typing import Optional, Tuple

class FloatQuantizePlan:
    def __init__(self, bits: int, exponent_bits: int, mantissa_bits: int, 
                 bias: Optional[int] = None, has_denormal: bool = True):
        """浮点数量化方案"""
        self.bits = bits
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        
        self.sign_bits = 1
        assert bits == 1 + exponent_bits + mantissa_bits, \
            f"Total bits {bits} != S1 + E{exponent_bits} + M{mantissa_bits}"
        
        # 计算指数偏置 (Standard IEEE 754 style)
        if bias is None:
            self.bias = 2**(exponent_bits - 1) - 1
        else:
            self.bias = bias
        
        self.has_denormal = has_denormal
        
        # 内部表示的最大指数值 (全1)
        self.max_exp_codepoint = (1 << exponent_bits) - 1
        
        # 最大规格化数 (指数为 max_exp-1)
        # E_max = (2^E - 2) - bias
        max_exp_val = (self.max_exp_codepoint - 1) - self.bias
        self.max_normal = (2 - 2**(-mantissa_bits)) * 2**max_exp_val
        
        # 最小规格化数
        self.min_normal = 2**(1 - self.bias)
        
        # 最小非规格化数
        self.min_denormal = 2**(1 - self.bias - mantissa_bits) if has_denormal else self.min_normal
        
        self.max_value = self.max_normal
    
    def __repr__(self):
        return f"FloatQuantizePlan(bits={self.bits}, E{self.exponent_bits}M{self.mantissa_bits}, bias={self.bias})"

class Quantizer:
    def quantize(self, x: np.ndarray, normalized_range: float = 1.0) -> np.ndarray:
        raise NotImplementedError
    
    def dequantize(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class FloatQuantizer(Quantizer):
    def __init__(self, plan: FloatQuantizePlan):
        self.plan = plan
        self.exp_max_val = (1 << self.plan.exponent_bits) - 2 # Max valid normal exponent code
        self.exp_inf_nan = (1 << self.plan.exponent_bits) - 1
    
    def quantize(self, x: np.ndarray, normalized_range: float = 1.0) -> np.ndarray:
        """量化浮点数数组"""
        # 1. 缩放
        # 避免除以0，如果是0范围则不缩放
        scale = self.plan.max_value / normalized_range if normalized_range > 0 else 1.0
        x_scaled = x * scale
        
        # 2. 提取符号
        sign = (x_scaled < 0).astype(np.uint32)
        abs_x = np.abs(x_scaled)
        
        # 3. 使用 frexp 分解: x = mant * 2^exp, where mant in [0.5, 1)
        # 注意：IEEE标准通常假设 mant in [1.0, 2.0)。
        # np.frexp 返回的 mantissa 是 0.5~1.0，exponent 是相应的。
        # 例如 1.5 -> mant=0.75, exp=1 (0.75 * 2^1 = 1.5)
        # 我们需要将其转换为 1.X * 2^Y 的形式。
        mant, exp = np.frexp(abs_x)
        
        # 将 [0.5, 1) 映射到 [1.0, 2.0) 
        # m' = m * 2, e' = e - 1
        # 但要注意处理 0 的情况
        mask_nonzero = (abs_x != 0)
        mant[mask_nonzero] *= 2
        exp[mask_nonzero] -= 1
        
        # 此时：abs_x = mant * 2^exp, 其中 mant [1.0, 2.0)
        # 目标指数: E_biased = exp + bias
        e_biased = exp.astype(np.int32) + self.plan.bias
        
        # --- 处理非规格化数 (Subnormal) ---
        if self.plan.has_denormal:
            # 规格化数的最小 biased exponent 是 1
            # 如果 e_biased < 1，说明是 Subnormal
            # 标准做法：将指数固定为 1-bias (即编码为0)，相应调整尾数
            # 现在的 exp 是真实指数。
            # Subnormal 目标: val = 0.mant * 2^(1-bias)
            # 当前: val = 1.mant * 2^exp
            # shift = (1 - bias) - exp
            # new_mant = 1.mant / 2^shift
            
            subnormal_mask = (e_biased < 1) & mask_nonzero
            if np.any(subnormal_mask):
                shift = (1 - self.plan.bias) - exp[subnormal_mask]
                mant[subnormal_mask] /= (2.0 ** shift)
                e_biased[subnormal_mask] = 0 # 编码为0
        
        # --- 量化尾数 ---
        # 规格化数：去掉整数部分1，保留小数部分。非规格化数：直接保留。
        # 我们这里统一处理：先计算除去隐含位后的值，然后 round
        
        # 规格化数掩码 (e_biased >= 1)
        normal_mask = (e_biased >= 1) & mask_nonzero
        
        # 映射到整数域 [0, 2^M]
        # 对于 Normal: (mant - 1) * 2^M
        # 对于 Subnormal: mant * 2^M (因为Subnormal没有隐含的1)
        
        mant_int = np.zeros_like(mant)
        
        if np.any(normal_mask):
            mant_int[normal_mask] = (mant[normal_mask] - 1.0) * (2 ** self.plan.mantissa_bits)
            
        if self.plan.has_denormal and np.any(subnormal_mask):
            mant_int[subnormal_mask] = mant[subnormal_mask] * (2 ** self.plan.mantissa_bits)
            
        # 四舍五入
        mant_int_rounded = np.round(mant_int)
        
        # --- 关键修复：处理进位 (Rounding Overflow) ---
        # 如果 mant_int_rounded == 2^M，说明进位了 (例如 1.99 -> 2.0)
        # 对于 Normal，这意味着指数 +1，尾数变为 0
        # 对于 Subnormal，这意味着可能变成了 Normal 最小数
        
        overflow_mask = (mant_int_rounded >= (2 ** self.plan.mantissa_bits))
        if np.any(overflow_mask):
            # 尾数归零
            mant_int_rounded[overflow_mask] = 0
            # 指数加1
            e_biased[overflow_mask] += 1
            # 注意：如果本来是 Subnormal (e=0)，进位后变成 Normal (e=1)，逻辑也是自洽的
            
        # --- 处理指数溢出 (Clamp to Max Finite) ---
        # 如果 e_biased > max_exp，设为最大有限数 (Inf处理比较复杂，这里简化为Clamp)
        # 或者设为 Inf (e=max_exp_codepoint, m=0)
        # 这里为了模型量化通常做 Clamp
        
        clamp_mask = (e_biased > self.exp_max_val)
        if np.any(clamp_mask):
            e_biased[clamp_mask] = self.exp_max_val
            mant_int_rounded[clamp_mask] = (1 << self.plan.mantissa_bits) - 1 # Max mantissa
            
        # 处理 0 (保留符号位是高级特性，这里简单处理)
        zero_mask = (abs_x == 0)
        e_biased[zero_mask] = 0
        mant_int_rounded[zero_mask] = 0
        
        # 组合整数
        q_sign = sign.astype(np.uint32) << (self.plan.exponent_bits + self.plan.mantissa_bits)
        q_exp = e_biased.astype(np.uint32) << self.plan.mantissa_bits
        q_mant = mant_int_rounded.astype(np.uint32)
        
        result = q_sign | q_exp | q_mant
        return result
    
    def dequantize(self, quantized: np.ndarray, normalized_range: float = 1.0) -> np.ndarray:
        """反量化"""
        # 提取掩码
        M = self.plan.mantissa_bits
        E = self.plan.exponent_bits
        
        # 提取分量
        s = (quantized >> (E + M)) & 1
        e = (quantized >> M) & ((1 << E) - 1)
        m = quantized & ((1 << M) - 1)
        
        # 转换为浮点计算
        m_float = m.astype(np.float32)
        
        # 结果容器
        val = np.zeros_like(quantized, dtype=np.float32)
        
        # 1. 零和非规格化数 (e == 0)
        if self.plan.has_denormal:
            subnormal_mask = (e == 0)
            # val = (-1)^s * 2^(1-bias) * (m / 2^M)
            if np.any(subnormal_mask):
                mant_val = m_float[subnormal_mask] / (2**M)
                val[subnormal_mask] = mant_val * (2**(1 - self.plan.bias))
        else:
            # 如果不支持非规格化，e=0通常代表0
            pass 
            
        # 2. 规格化数 (0 < e < max)
        normal_mask = (e > 0) & (e < ((1 << E) - 1))
        if np.any(normal_mask):
            # val = (-1)^s * 2^(e-bias) * (1 + m / 2^M)
            exponent = e[normal_mask].astype(np.int32) - self.plan.bias
            mant_val = 1.0 + m_float[normal_mask] / (2**M)
            val[normal_mask] = mant_val * (2.0 ** exponent)
            
        # 3. Inf / NaN (e == max) - 简化处理，全视作最大值
        inf_mask = (e == ((1 << E) - 1))
        if np.any(inf_mask):
            val[inf_mask] = np.inf # 或者 max_value
            
        # 应用符号
        val = val * ((-1)**s)
        
        # 反缩放
        scale = normalized_range / self.plan.max_value if self.plan.max_value > 0 else 1.0
        return val * scale

class Int2BitSpecialQuantizer:
    def __init__(self):
        self.bits = 2
        
    def quantize(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.uint8)
        abs_x = np.abs(x)
        
        # 1. 定义统一的死区
        zero_mask = abs_x < 0.1
        
        # 2. 对称映射逻辑
        # 我们重新分配索引：0=0, 1=正台阶, 2=负台阶, 3=备用
        
        # 修改后的 4 态对称方案：
        mask_small = (~zero_mask) & (abs_x <= 0.5946)
        mask_large = (~zero_mask) & (abs_x > 0.5946)
        
        out[(~zero_mask) & (x > 0)] = 1 # 正
        out[(~zero_mask) & (x < 0)] = 2 # 负
        
        return out

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        # 索引: 0    1 (正)         2 (负)         3 (备用)
        # 为了能量平衡，正负代表值必须绝对值相等
        val = 0.5 # 这是一个经验值，可以根据听感调整
        lookup = [0.0, val, -val, 0.0]
        return np.take(lookup, q).astype(np.float32)

def float_pack(bits: int, q: np.ndarray) -> bytes:
    """将量化后的整数数组打包为字节流"""
    q = q.astype(np.uint8) # 确保输入是 uint8
    
    if bits == 8:
        return q.tobytes()
        
    elif bits == 6:
        # Pack 4 fp6 numbers (4 * 6 = 24 bits) into 3 bytes
        # 输入形状必须能被4整除
        n = q.size
        assert n % 4 == 0, "Size must be divisible by 4 for FP6 packing"
        
        # Reshape to (N/4, 4)
        q_reshaped = q.reshape(-1, 4)
        
        out = np.zeros((q_reshaped.shape[0], 3), dtype=np.uint8)
        
        # Byte 0: [v0: 6bits] | [v1: low 2bits]
        out[:, 0] = (q_reshaped[:, 0] & 0x3F) | ((q_reshaped[:, 1] & 0x03) << 6)
        
        # Byte 1: [v1: high 4bits] | [v2: low 4bits]
        out[:, 1] = ((q_reshaped[:, 1] >> 2) & 0x0F) | ((q_reshaped[:, 2] & 0x0F) << 4)
        
        # Byte 2: [v2: high 2bits] | [v3: 6bits]
        out[:, 2] = ((q_reshaped[:, 2] >> 4) & 0x03) | ((q_reshaped[:, 3] & 0x3F) << 2)
        
        return out.tobytes()
        
    elif bits == 4:
        # Pack 2 fp4 numbers into 1 byte
        n = q.size
        assert n % 2 == 0, "Size must be divisible by 2 for FP4 packing"
        
        q_reshaped = q.reshape(-1, 2)
        out = np.zeros((q_reshaped.shape[0], 1), dtype=np.uint8)
        
        # Low nibble: v0, High nibble: v1
        out[:, 0] = (q_reshaped[:, 0] & 0x0F) | ((q_reshaped[:, 1] & 0x0F) << 4)
        
        return out.tobytes()
    elif bits == 2:
        # Pack 4 numbers into 1 byte (2 * 4 = 8 bits)
        n = q.size
        assert n % 4 == 0, "Size must be divisible by 4 for FP2 packing"
        
        q_reshaped = q.reshape(-1, 4)
        out = np.zeros((q_reshaped.shape[0], 1), dtype=np.uint8)
        
        # 将 4 个 2-bit 值依次放入字节的位中
        # v0: bits 0-1, v1: bits 2-3, v2: bits 4-5, v3: bits 6-7
        out[:, 0] = (q_reshaped[:, 0] & 0x03) | \
                    ((q_reshaped[:, 1] & 0x03) << 2) | \
                    ((q_reshaped[:, 2] & 0x03) << 4) | \
                    ((q_reshaped[:, 3] & 0x03) << 6)
        
        return out.tobytes()
    else:
        raise NotImplementedError(f"Packing for {bits} bits not implemented")

def float_unpack(bits: int, data_bytes: bytes, count: int) -> np.ndarray:
    """从字节流解包"""
    # 将字节转为 uint8 数组
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    
    if bits == 8:
        return raw.astype(np.uint32)
        
    elif bits == 6:
        # 3 bytes -> 4 fp6
        n_groups = raw.size // 3
        raw = raw[:n_groups*3].reshape(-1, 3)
        out = np.zeros((n_groups, 4), dtype=np.uint8)
        
        b0, b1, b2 = raw[:, 0], raw[:, 1], raw[:, 2]
        
        # v0: B0 low 6 bits
        out[:, 0] = b0 & 0x3F
        # v1: B0 high 2 bits | B1 low 4 bits
        out[:, 1] = (b0 >> 6) | ((b1 & 0x0F) << 2)
        # v2: B1 high 4 bits | B2 low 2 bits
        out[:, 2] = (b1 >> 4) | ((b2 & 0x03) << 4)
        # v3: B2 high 6 bits
        out[:, 3] = (b2 >> 2) & 0x3F
        
        return out.flatten()[:count] # 截取到请求的数量
        
    elif bits == 4:
        # 1 byte -> 2 fp4
        raw = raw.reshape(-1, 1)
        out = np.zeros((raw.shape[0], 2), dtype=np.uint8)
        
        out[:, 0] = raw[:, 0] & 0x0F
        out[:, 1] = (raw[:, 0] >> 4) & 0x0F
        
        return out.flatten()[:count]

    elif bits == 2:
        # 1 byte -> 4 numbers
        # 假设输入数据长度正确
        raw = raw.reshape(-1, 1)
        out = np.zeros((raw.shape[0], 4), dtype=np.uint8)
        
        out[:, 0] = raw[:, 0] & 0x03
        out[:, 1] = (raw[:, 0] >> 2) & 0x03
        out[:, 2] = (raw[:, 0] >> 4) & 0x03
        out[:, 3] = (raw[:, 0] >> 6) & 0x03
        
        return out.flatten()[:count]

    return np.array([])

if __name__ == "__main__":
    # Test
    quantizer = Int2BitSpecialQuantizer()
    data = 2 ** (np.random.rand(48) * 10 - 10) * np.sign(np.random.rand(48) - 0.5)
    quantized = quantizer.quantize(data)
    packed = float_pack(2, quantized)
    unpacked = float_unpack(2, packed, quantized.size)
    dequantized = quantizer.dequantize(unpacked)
    print(data,quantized,"\n", dequantized)