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

QUANT_LEVELS = {
    # 8bit float
    0:  {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8,4,3))},
    1:  {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8,5,2))},
    2:  {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8,3,4))},

    # 6bit float
    3:  {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6,4,1))},
    4:  {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6,3,2))},
    5:  {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6,2,3))},

    # 4bit float
    6:  {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4,3,0))},
    7:  {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4,2,1))},
    8:  {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4,1,2))},

    # int plans
    9:  {'bits': 2, 'plan': Int2BitSpecialQuantizer(IntQuantizePlan(2))},

    # empty
    15: {'bits': 0, 'plan': None}
}

class QFP3Codec:
    
    def __init__(self):
        self.version = 3
        self.magic = b"QFPA"
    
    def compress(self, audio_file, out_file, qp = 32):
        data, sample_rate = sf.read(audio_file)
        win_size = round_power_2(sample_rate / 8)
        group_size = 64

        if len(data.shape) != 2 or data.shape[1] != 2:
            # copy data to stereo if it's mono
            if len(data.shape) == 1:
                data = np.column_stack((data, data))
            else:
                raise ValueError("Input audio must be mono or stereo (2 channels).")
        
        data = data[:sample_rate*15]  # 截取前30秒
        hop_size = win_size // 2
        num_windows = (len(data) // hop_size) - 1

        nyquist = sample_rate // 2
        
        # 填充data到win_size的整数倍
        if len(data) % win_size != 0:
            padding_size = win_size - (len(data) % win_size)
            data = np.pad(data, ((0, padding_size), (0, 0)), mode='constant')
        
        if len(data.shape) != 2 or data.shape[1] != 2:
            raise ValueError("Input audio must be stereo (2 channels).")
        
        with open(out_file, "wb") as f:
            f.write(self.magic)
            f.write(struct.pack("B", self.version))
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", win_size))
            f.write(struct.pack("<I", num_windows))
            f.write(struct.pack("<I", group_size))
            
            
            window = create_mdct_window(win_size)

            io = BytesIO()

            
            band_plan =[]
            num_bands = hop_size // group_size

            for i in range(0, hop_size, group_size):
                min_freq = int(i * nyquist / hop_size)
                max_freq = int((i + group_size) * nyquist / hop_size)
                if max_freq < 4000:
                    quantizer_level = 0
                    T = 0.2
                elif max_freq < 8000:
                    quantizer_level = 3
                    T = 1
                elif max_freq < 10000:
                    quantizer_level = 5
                    T = 6
                elif max_freq < 12000:
                    quantizer_level = 6
                    T = 9
                elif max_freq < 14000:
                    quantizer_level = 7
                    T = 13
                elif max_freq < 18000:
                    quantizer_level = 9
                    T = 16
                elif max_freq > 20000:
                    quantizer_level = 15
                else:
                    quantizer_level = 9
                    T = 19
                band_plan.append(
                    {
                        "quantizer_level": quantizer_level,
                        "T": T
                    }
                )

            # 构造 L+R 的 Plan 数组
            lpr_plan_arr = np.array([band['quantizer_level'] for band in band_plan], dtype=np.uint8)

            # 构造 L-R 的 Plan 数组 (应用你逻辑中的 +3 偏移)
            lmr_plan_arr = np.array([
                np.clip(lvl + 3, 0, 9) if lvl < 9 else lvl 
                for lvl in lpr_plan_arr
            ], dtype=np.uint8)

            quant_levels = [
                lpr_plan_arr, lmr_plan_arr
            ]

            # 写入两个 Plan (每个 4-bit 打包)
            f.write(float_pack(4, lpr_plan_arr))
            f.write(float_pack(4, lmr_plan_arr))
                            
            
            for i in tqdm(range(num_windows), desc="压缩进度"):
            
                # Process both L+R and L-R channels
                L = data[i * hop_size : i * hop_size + win_size, 0]
                R = data[i * hop_size : i * hop_size + win_size, 1]

                LPR = (L + R)* window
                LMR = (L - R)* window

                frame_bytes = bytearray()

                quantizers = {}
                silent = True

                # 2. MDCT 变换
                for ch, block in enumerate([LPR, LMR]):
                    mdct_coeff = mdct(block) / win_size

                    bands = []

                    band_bitmap=np.zeros(num_bands, dtype=np.uint8)

                    for i in range(0, hop_size, group_size):
                        band_idx = i // group_size
                        quantizer_level = quant_levels[ch][band_idx]
                        T = band_plan[band_idx]['T']

                        scale = np.max(np.abs(mdct_coeff[i:i+group_size]))
                        if scale == 0:
                            scale = 1

                        mdct_coeff[i:i+group_size] = mdct_coeff[i:i+group_size] / scale

                        threshold = (qp / 63) * 0.1 * T
                        mdct_coeff[i:i+group_size] = np.where(np.abs(mdct_coeff[i:i+group_size]) < threshold, 0, mdct_coeff[i:i+group_size])

                        quantizer = QUANT_LEVELS[quantizer_level]['plan']
                        if quantizer is None:
                            continue
                        quantized = quantizer.quantize(mdct_coeff[i:i+group_size])

                        # 检测是否全0
                        if np.all(quantized == 0):
                            continue
                        else:

                            pos = pos_encode(quantized, min_length=3*8//quantizer.plan.bits)
                            pos_bytes = encode_pos_arr(pos['pos'])
                            stripped = np.array(pos['stripped'])

                            # 3. 补齐 stripped 数组以符合打包器的整除要求
                            # 不同 bits 对应的对齐要求
                            align_requirement = {8: 1, 6: 4, 4: 2, 2: 4, 1: 8}
                            required_mod = align_requirement.get(quantizer.plan.bits, 1)

                            remainder = len(stripped) % required_mod
                            if remainder != 0:
                                padding_size = required_mod - remainder
                                # 补 0 是安全的，因为解包时我们会根据 original_stripped_len 截断
                                stripped = np.concatenate([stripped, np.zeros(padding_size, dtype=np.uint8)])

                            result = float_pack(quantizer.plan.bits, stripped)
                            band_bitmap[band_idx] = 1
                            silent = False
                            band = {
                                "scale": scale,
                                "pos_bytes": pos_bytes,
                                "result": result
                            }
                            bands.append(band)

                    # Write band bitmap
                    bitmap = np.packbits(band_bitmap)
                    print(bitmap)
                    frame_bytes.extend(bitmap.tobytes())

                    for band_idx, band in enumerate(bands):
                        if band_bitmap[band_idx] == 0:
                            continue
                        q_s = encode_scale(band['scale'])
                        frame_bytes.append(q_s) # 直接 append 一个 byte
                        frame_bytes.extend(prefix_encode(len(band['pos_bytes'])))
                        frame_bytes.extend(band['pos_bytes'])
                        frame_bytes.extend(prefix_encode(len(band['result'])))
                        frame_bytes.extend(band['result'])

                # 全是静音，直接跳过
                if silent:
                    io.write(encapsulate_frame(b''))
                    continue

                frame_bytes.extend(
                    struct.pack("<H", crc16(bytes(frame_bytes)))
                )

                io.write(encapsulate_frame(frame_bytes))


            qfp_data = io.getvalue()
            compressed_data = zlib.compress(qfp_data)
            f.write(compressed_data)


    def decompress(self, in_file, out_file):
        with open(in_file, "rb") as f:
            magic = f.read(4)
            if magic != self.magic:
                raise ValueError("Invalid file format")
            
            version = struct.unpack("B", f.read(1))[0]
            if version != self.version:
                raise ValueError(f"Unsupported version: {version}")
            
            sample_rate = struct.unpack("<I", f.read(4))[0]
            win_size = struct.unpack("<I", f.read(4))[0]
            num_windows = struct.unpack("<I", f.read(4))[0]
            group_size = struct.unpack("<I", f.read(4))[0]

            hop_size = win_size // 2
            window = create_mdct_window(win_size)
            nyquist = sample_rate // 2

            num_bands = hop_size // group_size
            plan_bytes_per_ch = (num_bands + 1) // 2

            # 依次读取两个通道的 Plan   
            lpr_plan = float_unpack(4, f.read(plan_bytes_per_ch), num_bands)
            lmr_plan = float_unpack(4, f.read(plan_bytes_per_ch), num_bands)
            dual_ch_plans = [lpr_plan, lmr_plan] # 放入列表方便循环调用
            
            recon_audio = []
            

            out_buffer = np.zeros((2, hop_size*3), dtype=np.float32)
            print(f"读取压缩数据 (版本 {version}, 采样率 {sample_rate}, 窗口大小 {win_size}, 窗口数 {num_windows})")

            data = zlib.decompress(f.read())
            
            io = BytesIO(data)

            recon = np.zeros((2, win_size), dtype=np.float32)
            
            with open('out.dump', 'wb') as f:
                f.write(data)


            br_diagnosis = {
                "band_plan": 0,
                "pos_bytes": 0,
                "result": 0
            }

            
            for _ in tqdm(range(num_windows), desc="解压进度"): 
                
                frame_bytes = decapsulate_frame(io)
                if len(frame_bytes) == 0:
                    # 空帧，直接添加hopsize个0到音频
                    recon_audio.append(np.column_stack((np.zeros(hop_size), np.zeros(hop_size))))
                    continue
                frame = BytesIO(frame_bytes)
                
                
                for i in range(2):

                    mdct_coeff = np.zeros(hop_size+group_size)
                    br_diagnosis["band_plan"] += num_bands // 2

                    bitmap_size = np.ceil(num_bands / 8)
                    band_bitmap = np.frombuffer(frame.read(int(bitmap_size)), dtype=np.uint8)
                    band_bitmap = np.unpackbits(band_bitmap)


                    for j in range(0, hop_size, group_size):
                        band_idx = j // group_size
                        if band_bitmap[band_idx] == 0:
                            continue
                        quantizer_level = dual_ch_plans[i][band_idx]
                        q_s = frame.read(1)[0]
                        scale = decode_scale(q_s)
                        pos_length = prefix_decode(frame)
                        pos = decode_pos_arr(BytesIO(frame.read(pos_length)))
                        length = prefix_decode(frame)
                        quantized = frame.read(length)

                        br_diagnosis["pos_bytes"] += pos_length
                        br_diagnosis["result"] += length
                        
                        quantizer = QUANT_LEVELS[quantizer_level]['plan']
                        stripped = float_unpack(quantizer.plan.bits, quantized, length * 8 // quantizer.plan.bits)
                        
                        quantized = np.array(pos_decode(pos, stripped, group_size))[:group_size]
                        mdct_coeff[j:j+group_size] = quantizer.dequantize(quantized) * scale
                        print(quantized)

                    recon[i] = imdct(mdct_coeff[:hop_size] * win_size) * window

                crc_frame = struct.unpack("<H", frame.read(2))[0]
                if crc16(frame_bytes[:-2]) != crc_frame:
                    raise ValueError("CRC check failed")

                print(br_diagnosis)

                LPR, LMR = recon
                
                # 将重构的信号添加到输出缓冲区
                out_buffer[0, hop_size:] += LPR
                out_buffer[1, hop_size:] += LMR

                # 计算 L 和 R 信号
                L = (out_buffer[0, :hop_size] + out_buffer[1, :hop_size]) / 2
                R = (out_buffer[0, :hop_size] - out_buffer[1, :hop_size]) / 2
                
                # 将 L 和 R 信号添加到数据列表
                recon_audio.append(np.column_stack((L, R)))

                # 前进缓冲区
                out_buffer = np.roll(out_buffer, -hop_size, axis=1)

                # 腾出的空间写0
                out_buffer[:, -hop_size:] = 0
                
            
            
            # 合并所有窗口的数据
            audio_data = np.concatenate(recon_audio, axis=0)
            
            # 保存音频文件
            sf.write(out_file, audio_data, sample_rate)

if __name__ == "__main__":
    encoder = QFP3Codec()
    encoder.compress("music.flac", "test.qfp")
    encoder.decompress("test.qfp", "result.wav")