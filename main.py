import soundfile as sf
import numpy as np
from dsp import mdct, imdct, create_mdct_window
from util import round_power_2
from quant_plan import FloatQuantizePlan, FloatQuantizer, float_pack, float_unpack, Int2BitSpecialQuantizer, IntQuantizePlan
from num_encoder import prefix_encode, prefix_decode, pos_encode, pos_decode, encode_pos_arr, decode_pos_arr, encode_scale, decode_scale
import struct
from io import BytesIO
from tqdm import tqdm
import zlib
from frame import crc16, encapsulate_frame, decapsulate_frame

# --- 1. 定义量化级别映射表 ---
# 包含 8-bit, 6-bit, 4-bit 浮点量化以及 2-bit 特殊整数量化
QUANT_LEVELS = {
    0: {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8,4,3))},
    1: {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8,5,2))},
    2: {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8,3,4))},
    3: {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6,4,1))},
    4: {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6,3,2))},
    5: {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6,2,3))},
    6: {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4,3,0))},
    7: {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4,2,1))},
    8: {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4,1,2))},
    9: {'bits': 2, 'plan': Int2BitSpecialQuantizer(IntQuantizePlan(2))},
    15: {'bits': 0, 'plan': None}
}

class QFP3Codec:
    def __init__(self):
        self.version = 3
        self.magic = b"QFPA"
    
    def compress(self, audio_file, out_file, qp=32):
        # --- 步骤 1: 预处理与参数初始化 ---
        data, sample_rate = sf.read(audio_file)
        win_size = round_power_2(sample_rate / 8)
        group_size = 64
        hop_size = win_size // 2
        nyquist = sample_rate // 2
        num_bands = hop_size // group_size

        # 强制立体声转换与填充
        if len(data.shape) == 1:
            data = np.column_stack((data, data))
        data = data[:sample_rate * 15] # 截取 15 秒
        if len(data) % win_size != 0:
            padding_size = win_size - (len(data) % win_size)
            data = np.pad(data, ((0, padding_size), (0, 0)), mode='constant')
        
        num_windows = (len(data) // hop_size) - 1
        window = create_mdct_window(win_size)

        # --- 步骤 2: 构建频率自适应 Band Plan (L+R 通道) ---
        band_plan = []
        for i in range(0, hop_size, group_size):
            max_freq = int((i + group_size) * nyquist / hop_size)
            # 根据频率设定量化级别与死区阈值 T
            if max_freq < 4000:   quantizer_level, T = 0, 0.2
            elif max_freq < 8000: quantizer_level, T = 3, 1
            elif max_freq < 10000: quantizer_level, T = 5, 6
            elif max_freq < 12000: quantizer_level, T = 6, 9
            elif max_freq < 14000: quantizer_level, T = 7, 13
            elif max_freq < 18000: quantizer_level, T = 9, 16
            elif max_freq > 20000: quantizer_level, T = 15, 0
            else:                  quantizer_level, T = 9, 19
            band_plan.append({"quantizer_level": quantizer_level, "T": T})

        # --- 步骤 3: 构造并保存双通道静态 Plan ---
        lpr_plan_arr = np.array([b['quantizer_level'] for b in band_plan], dtype=np.uint8)
        lmr_plan_arr = np.array([np.clip(lvl + 3, 0, 9) if lvl < 9 else lvl for lvl in lpr_plan_arr], dtype=np.uint8)
        quant_levels_cache = [lpr_plan_arr, lmr_plan_arr]

        with open(out_file, "wb") as f:
            # 写入文件头
            f.write(self.magic + struct.pack("B", self.version))
            f.write(struct.pack("<IIII", sample_rate, win_size, num_windows, group_size))
            # 写入 4-bit 打包后的双通道 Plan
            f.write(float_pack(4, lpr_plan_arr))
            f.write(float_pack(4, lmr_plan_arr))
            
            # --- 步骤 4: 核心帧压缩循环 ---
            io = BytesIO()
            for i in tqdm(range(num_windows), desc="压缩进度"):
                L = data[i * hop_size : i * hop_size + win_size, 0]
                R = data[i * hop_size : i * hop_size + win_size, 1]
                LPR, LMR = (L + R) * window, (L - R) * window
                
                frame_bytes = bytearray()
                silent = True

                for ch, block in enumerate([LPR, LMR]):
                    mdct_coeff = mdct(block) / win_size
                    bands_data = []
                    band_bitmap = np.zeros(num_bands, dtype=np.uint8)

                    for b_idx in range(num_bands):
                        start = b_idx * group_size
                        q_lvl = quant_levels_cache[ch][b_idx]
                        T_val = band_plan[b_idx]['T']
                        
                        # 归一化与死区裁剪
                        block_data = mdct_coeff[start:start+group_size]
                        scale = np.max(np.abs(block_data)) if np.max(np.abs(block_data)) > 0 else 1
                        norm_data = block_data / scale
                        threshold = (qp / 63) * 0.1 * T_val
                        norm_data = np.where(np.abs(norm_data) < threshold, 0, norm_data)

                        quantizer = QUANT_LEVELS[q_lvl]['plan']
                        if quantizer is None: continue
                        
                        quantized = quantizer.quantize(norm_data)
                        if np.all(quantized == 0): continue
                        
                        # 位置编码与位对齐打包
                        pos_info = pos_encode(quantized, min_length=3*8//quantizer.plan.bits)
                        stripped = np.array(pos_info['stripped'])
                        
                        align = {8: 1, 6: 4, 4: 2, 2: 4, 1: 8}.get(quantizer.plan.bits, 1)
                        if len(stripped) % align != 0:
                            stripped = np.concatenate([stripped, np.zeros(align - (len(stripped) % align), dtype=np.uint8)])

                        # 记录 Band 数据
                        band_bitmap[b_idx] = 1
                        silent = False
                        bands_data.append({
                            "scale": scale,
                            "pos_bytes": encode_pos_arr(pos_info['pos']),
                            "result": float_pack(quantizer.plan.bits, stripped)
                        })

                    # 写入通道 Bitmap
                    frame_bytes.extend(np.packbits(band_bitmap).tobytes())
                    # 写入通道各 Band 详细数据
                    for b in bands_data:
                        frame_bytes.append(encode_scale(b['scale']))
                        frame_bytes.extend(prefix_encode(len(b['pos_bytes'])) + b['pos_bytes'])
                        frame_bytes.extend(prefix_encode(len(b['result'])) + b['result'])

                # 封装帧并计算 CRC
                if silent:
                    io.write(encapsulate_frame(b''))
                else:
                    frame_bytes.extend(struct.pack("<H", crc16(bytes(frame_bytes))))
                    io.write(encapsulate_frame(frame_bytes))

            # zlib 终极压缩
            f.write(zlib.compress(io.getvalue()))

    def decompress(self, in_file, out_file):
        with open(in_file, "rb") as f:
            # --- 步骤 5: 读取 Header 与双通道 Plan ---
            magic, version = f.read(4), struct.unpack("B", f.read(1))[0]
            sr, win_size, num_windows, group_size = struct.unpack("<IIII", f.read(16))
            
            hop_size = win_size // 2
            num_bands = hop_size // group_size
            plan_bytes_len = (num_bands + 1) // 2
            
            # 加载双通道静态 Plan 查表
            lpr_plan = float_unpack(4, f.read(plan_bytes_len), num_bands)
            lmr_plan = float_unpack(4, f.read(plan_bytes_len), num_bands)
            dual_ch_plans = [lpr_plan, lmr_plan]
            
            data = zlib.decompress(f.read())
            io = BytesIO(data)
            window = create_mdct_window(win_size)
            recon_audio = []
            out_buffer = np.zeros((2, hop_size * 3), dtype=np.float32)

            # --- 步骤 6: 解压帧处理循环 ---
            for _ in tqdm(range(num_windows), desc="解压进度"):
                frame_bytes = decapsulate_frame(io)
                if len(frame_bytes) == 0:
                    recon_audio.append(np.zeros((hop_size, 2)))
                    continue
                
                frame = BytesIO(frame_bytes)
                ch_recon = np.zeros((2, win_size), dtype=np.float32)
                
                for i in range(2):
                    mdct_coeff = np.zeros(hop_size)
                    bitmap_size = int(np.ceil(num_bands / 8))
                    band_bitmap = np.unpackbits(np.frombuffer(frame.read(bitmap_size), dtype=np.uint8))

                    for j in range(0, hop_size, group_size):
                        band_idx = j // group_size
                        if band_bitmap[band_idx] == 0: continue
                        
                        # 依照 Plan 恢复量化器
                        q_lvl = dual_ch_plans[i][band_idx]
                        scale = decode_scale(frame.read(1)[0])
                        
                        # 恢复位置与 stripped 数据
                        pos_len = prefix_decode(frame)
                        pos = decode_pos_arr(BytesIO(frame.read(pos_len)))
                        res_len = prefix_decode(frame)
                        quant_data = frame.read(res_len)
                        
                        quantizer = QUANT_LEVELS[q_lvl]['plan']
                        bits = quantizer.plan.bits
                        stripped = float_unpack(bits, quant_data, res_len * 8 // bits)
                        
                        # 反量化并恢复 Band
                        quantized = np.array(pos_decode(pos, stripped, group_size))[:group_size]
                        mdct_coeff[j:j+group_size] = quantizer.dequantize(quantized) * scale

                    ch_recon[i] = imdct(mdct_coeff * win_size) * window

                # CRC 校验
                crc_stored = struct.unpack("<H", frame.read(2))[0]
                if crc16(frame_bytes[:-2]) != crc_stored:
                    raise ValueError("CRC check failed")

                # --- 步骤 7: L+R / L-R 还原与重叠相加 (OLA) ---
                LPR, LMR = ch_recon
                out_buffer[0, hop_size:] += LPR
                out_buffer[1, hop_size:] += LMR

                L_out = (out_buffer[0, :hop_size] + out_buffer[1, :hop_size]) / 2
                R_out = (out_buffer[0, :hop_size] - out_buffer[1, :hop_size]) / 2
                recon_audio.append(np.column_stack((L_out, R_out)))

                # 缓冲区平移
                out_buffer = np.roll(out_buffer, -hop_size, axis=1)
                out_buffer[:, -hop_size:] = 0
            
            # 合并写入音频
            sf.write(out_file, np.concatenate(recon_audio, axis=0), sr)

if __name__ == "__main__":
    encoder = QFP3Codec()
    encoder.compress("music.flac", "test.qfp")
    encoder.decompress("test.qfp", "result.wav")