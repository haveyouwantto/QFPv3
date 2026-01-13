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

def warp_steep(x, k=2):
    return (x**k) / (x**k + (1-x)**k)

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
        data = data[sample_rate * 8:sample_rate * 16] # 截取 15 秒
        if len(data) % win_size != 0:
            padding_size = win_size - (len(data) % win_size)
            data = np.pad(data, ((0, padding_size), (0, 0)), mode='constant')
        
        num_windows = (len(data) // hop_size) - 1
        window = create_mdct_window(win_size)

        # --- 步骤 2: 构建频率自适应 Band Plan (L+R 通道) ---
        band_plan = []
        for i in range(0, hop_size, group_size):
            max_freq = int((i + group_size) * nyquist / hop_size)
            # 设定基础保留比例：低频几乎全保，高频大幅削减
            if max_freq < 4000:    quantizer_level, kr_base = 0, 2
            elif max_freq < 6000:  quantizer_level, kr_base = 3, 1
            elif max_freq < 8000:  quantizer_level, kr_base = 5, 1/2
            elif max_freq < 10000: quantizer_level, kr_base = 6, 1/4
            elif max_freq < 12000: quantizer_level, kr_base = 7, 1/8
            elif max_freq < 14000: quantizer_level, kr_base = 8, 1/16
            elif max_freq < 16000: quantizer_level, kr_base = 9, 1/32
            elif max_freq < 18000: quantizer_level, kr_base = 9, 1/64
            elif max_freq > 20000: quantizer_level, kr_base = 15, 0.0
            else:                  quantizer_level, kr_base = 9, 1/128
            
            band_plan.append({"quantizer_level": quantizer_level, "kr_base": kr_base})
            

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

                # 计算 L+R 和 L-R 的总能量 (或者用绝对值之和)
                energy_lpr = np.sum(np.abs(LPR))
                energy_lmr = np.sum(np.abs(LMR))

                # 计算差异系数 (0.0 ~ 1.0)
                # 如果 energy_lpr 很大而 energy_lmr 很小，ms_ratio 会接近 0
                ms_ratio = energy_lmr / (energy_lpr + 1e-6)
                
                frame_bytes = bytearray()
                silent = True

                quantize_bits = [8,6,4,2]

                for ch, block in enumerate([LPR, LMR]):
                    mdct_coeff = mdct(block) / win_size
                    bands_data = []
                    band_bitmap = np.zeros(num_bands, dtype=np.uint8)

                    mdct_buckets = {8: [], 6: [], 4: [], 2: []}
                    # print("="*40)

                    for b_idx in range(num_bands):
                        start = b_idx * group_size
                        q_lvl = quant_levels_cache[ch][b_idx]
                        q_bits = QUANT_LEVELS[q_lvl]['bits']

                        if q_lvl == 15: continue
                        
                        # 归一化
                        block_data = mdct_coeff[start:start+group_size]
                        scale = np.max(np.abs(block_data)) if np.max(np.abs(block_data)) > 0 else 1
                        norm_data = block_data / scale

                        # 比特削减

                        # 1. 收集当前通道所有 MDCT 系数的绝对值
                        all_coeffs = np.abs(norm_data)

                        # 2. 设定你的保留比例 (例如根据 QP 计算 ratio)
                        bit_bouns = (8 / q_bits)
                        channel_bouns = 1 if ch == 0 else np.clip(ms_ratio * 1.5, 0.1, 0.5)
                        keep_ratio = np.clip(
                            max(0, warp_steep(1.0 - (qp / 63), 1.5)) * band_plan[b_idx]['kr_base'] * bit_bouns * channel_bouns,
                            0,0.99
                        )
                        # # print(keep_ratio)
                        # if keep_ratio < 1 / group_size:
                        #     # 设置为全0
                        #     pruned_data = np.zeros_like(norm_data)
                        # else:
                        # 3. 找到截断阈值
                        # 使用 np.percentile 快速找到对应比例的阈值
                        threshold_val = np.percentile(all_coeffs, (1 - keep_ratio) * 100)

                        # 4. 执行“暴力”抹除
                        pruned_data = np.where(np.abs(norm_data) < threshold_val, 0, norm_data)

                        quantizer = QUANT_LEVELS[q_lvl]['plan']
                        if quantizer is None: continue
                        
                        quantized = quantizer.quantize(pruned_data)

                        # 损失能量计算
                        restored = quantizer.dequantize(quantized)

                        amp_loss = max(0, np.sum(np.abs(norm_data)) - np.sum(np.abs(restored)))
                        loss_byte = int(np.clip(amp_loss * 4, 0, 255))

                        # 记录 Band 数据
                        band_bitmap[b_idx] = 1
                        silent = False
                        bands_data.append({
                            "scale": scale,
                            "loss_byte": loss_byte
                        })
                        mdct_buckets[quantizer.plan.bits].extend(quantized)

                    # 写入通道 Bitmap
                    frame_bytes.extend(np.packbits(band_bitmap).tobytes())
                    # 写入通道各 Band 元数据
                    for b in bands_data:
                        frame_bytes.append(encode_scale(b['scale']))
                        frame_bytes.append(b['loss_byte']) # 写入 1 字节
                        
                    # 写入 MDCT 系数桶
                    for bits in quantize_bits:
                        bucket = mdct_buckets[bits]
                        pos_info = pos_encode(bucket, min_length=3*8//bits)
                        pos_bytes = encode_pos_arr(pos_info['pos'])
                        stripped = np.array(pos_info['stripped'])
                        align = {8: 1, 6: 4, 4: 2, 2: 4, 1: 8}.get(bits, 1)
                        if len(stripped) % align != 0:
                            stripped = np.concatenate([stripped, np.zeros(align - (len(stripped) % align), dtype=np.uint8)])
                        result = float_pack(bits, stripped)
                        frame_bytes.extend(prefix_encode(len(pos_bytes)) + pos_bytes)
                        frame_bytes.extend(prefix_encode(len(result)) + result)

                # 封装帧并计算 CRC
                if silent:
                    io.write(encapsulate_frame(b''))
                else:
                    crc = crc16(bytes(frame_bytes))
                    # 放在最前面
                    io.write(encapsulate_frame(struct.pack("<H", crc) + frame_bytes))

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
            with open("out.dump", "wb") as g:
                g.write(data)
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

                # CRC 校验
                crc_stored = struct.unpack("<H", frame.read(2))[0]
                if crc16(frame_bytes[2:]) != crc_stored:
                    raise ValueError("CRC check failed")
                
                for i in range(2): # 通道循环
                    mdct_coeff = np.zeros(hop_size)
                    bitmap_size = int(np.ceil(num_bands / 8))
                    band_bitmap = np.unpackbits(np.frombuffer(frame.read(bitmap_size), dtype=np.uint8))[:num_bands]
                    
                    # 1. 预读本通道所有 Band 的元数据 (Scale, Loss)
                    # 因为压缩时是按 band 顺序写的，所以这里也按顺序读
                    band_meta = []
                    for b_idx in range(num_bands):
                        if band_bitmap[b_idx] == 1:
                            scale = decode_scale(frame.read(1)[0])
                            loss_val = frame.read(1)[0] / 4.0
                            band_meta.append((scale, loss_val))
                    
                    # 2. 预读并恢复本通道的 4 个比特桶
                    # 每个桶恢复成一个大的一维 quantized 数组
                    buckets_quantized = {8: [], 6: [], 4: [], 2: []}
                    for bits in [8, 6, 4, 2]:
                        pos_len = prefix_decode(frame)
                        
                        pos = decode_pos_arr(BytesIO(frame.read(pos_len)))
                        res_len = prefix_decode(frame)
                        quant_data = frame.read(res_len)
                        if res_len == 0: continue # 该比特等级无数据
                        
                        # 难点：我们需要知道这个桶对应了多少个 Band，从而确定 total_length
                        # 统计本通道 bitmap 中，属于当前 bits 等级的 Band 数量
                        active_bands_count = 0
                        for b_idx in range(num_bands):
                            if band_bitmap[b_idx] == 1 and QUANT_LEVELS[dual_ch_plans[i][b_idx]]['bits'] == bits:
                                active_bands_count += 1

                        # 计算该桶中有多少个有效系数
                        stripped = float_unpack(bits, quant_data, len(quant_data) * 8 // bits)
                        
                        # 恢复出该桶完整的 quantized 序列
                        full_bucket = pos_decode(pos, stripped, active_bands_count * group_size)
                        buckets_quantized[bits] = np.array(full_bucket, dtype=np.uint8)

                    # 3. 分配数据：再次遍历 Band，从桶中“截稿”
                    meta_ptr = 0
                    bucket_ptrs = {8: 0, 6: 0, 4: 0, 2: 0} # 这里的指针以系数个数为单位
                    
                    amp_losses = np.zeros(num_bands)
                    
                    for b_idx in range(num_bands):
                        if band_bitmap[b_idx] == 0: continue
                        
                        q_bits = QUANT_LEVELS[dual_ch_plans[i][b_idx]]['bits']
                        scale, loss_norm = band_meta[meta_ptr]
                        meta_ptr += 1
                        
                        # 从对应 bits 的桶中截取 group_size 个系数
                        start_p = bucket_ptrs[q_bits]
                        end_p = start_p + group_size
                        q_slice = buckets_quantized[q_bits][start_p:end_p]
                        bucket_ptrs[q_bits] = end_p
                        if len(q_slice) == 0: continue
                        
                        # 反量化
                        quantizer = QUANT_LEVELS[dual_ch_plans[i][b_idx]]['plan']
                        mdct_coeff[b_idx*group_size : (b_idx+1)*group_size] = quantizer.dequantize(q_slice) * scale
                        amp_losses[b_idx] = loss_norm * scale

                    # 第二次遍历：执行噪声填充 (仿照 v2 的插值逻辑)
                    if np.sum(amp_losses) > 0:
                        # 构建插值点：每个 Band 的中心
                        x_old = np.arange(group_size // 2, hop_size, group_size)
                        # 获取有效的 loss 点（有数据的 band）
                        # 这里为了简单处理，全频段插值，无数据频段 loss 为 0
                        from scipy.interpolate import interp1d
                        
                        # 为防止边界突变，在两头补 0
                        x_points = np.concatenate([[-group_size], x_old, [hop_size + group_size]])
                        y_points = np.concatenate([[0], amp_losses, [0]])
                        
                        f_interp = interp1d(x_points, y_points, kind='linear')
                        # 生成全频段的噪声增益包络
                        noise_envelope = f_interp(np.arange(hop_size))
                        
                        # 生成原始噪声 (标准正态分布)
                        # 注意：sum(abs(noise)) 对于 group_size=64 约等于 64 * 0.8 = 51.2
                        # 我们需要根据 noise_envelope 控制这个 sum(abs)
                        raw_noise = np.random.normal(0, 1.0, hop_size)
                        
                        # 归一化每一组噪声，使其 sum(abs) 精确等于要求的 loss
                        for b_idx in range(num_bands):
                            idx_range = slice(b_idx * group_size, (b_idx + 1) * group_size)
                            curr_noise = raw_noise[idx_range]
                            curr_sum = np.sum(np.abs(curr_noise))
                            if curr_sum > 0:
                                # 将噪声注入到原系数为 0 的缝隙中
                                zero_mask = (mdct_coeff[idx_range] == 0)
                                # 强度调节：根据插值包络
                                target_amp = noise_envelope[idx_range] 
                                mdct_coeff[idx_range] += (curr_noise / curr_sum) * target_amp * zero_mask

                    ch_recon[i] = imdct(mdct_coeff * win_size) * window

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
    encoder.compress("寂れた浜辺(Deserted beach).wav", "test.qfp", qp=52)
    encoder.decompress("test.qfp", "result.wav")