"""
QFP3 音频编解码器 (Quantum Frequency Processor v3)
===================================================

这是一个基于 MDCT (Modified Discrete Cosine Transform) 的有损音频压缩算法。
主要特点：
  1. 使用 M/S (Mid/Side) 立体声编码技术
  2. 频率自适应量化 - 低频高精度,高频低精度
  3. 全局阈值剪枝 - 移除低能量系数
  4. 多比特桶分层压缩 - 8/6/4/2-bit 浮点量化
  5. 噪声填充技术 - 恢复被剪枝的频段能量

算法流程：
  编码: 音频 -> MDCT -> M/S 分离 -> 频段分割 -> 量化 -> 剪枝 -> 压缩
  解码: 解压 -> 反量化 -> 噪声填充 -> iMDCT -> M/S 合并 -> 重叠相加 (OLA)
"""

import soundfile as sf
import numpy as np
from dsp import mdct, imdct, create_mdct_window
from util import round_power_2
from quant_plan import (
    FloatQuantizePlan, FloatQuantizer, float_pack, float_unpack,
    Int2BitSpecialQuantizer
)
from num_encoder import (
    prefix_encode, prefix_decode, pos_encode, pos_decode,
    encode_pos_arr, decode_pos_arr, encode_scale, decode_scale
)
import struct
from io import BytesIO
from tqdm import tqdm
import zlib
from frame import crc16, encapsulate_frame, decapsulate_frame


# ============================================================================
# 量化级别配置
# ============================================================================

# 定义 10 个量化级别 (0-9) + 1 个跳过级别 (15)
# 每个级别包含：比特数、量化方案
# 
# 量化方案说明：
#   - FloatQuantizePlan(bits, exp_bits, mantissa_bits): 浮点量化
#     例如 (8,4,3) 表示 8-bit 量化,其中 4-bit 存储指数, 3-bit 存储尾数, 1-bit 符号
#   - IntQuantizePlan(bits): 整数量化 (仅用于 2-bit 特殊情况)
QUANT_LEVELS = {
    0:  {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8, 4, 3))},  # 最高精度
    1:  {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8, 5, 2))},
    2:  {'bits': 8, 'plan': FloatQuantizer(FloatQuantizePlan(8, 3, 4))},
    3:  {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6, 4, 1))},
    4:  {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6, 3, 2))},
    5:  {'bits': 6, 'plan': FloatQuantizer(FloatQuantizePlan(6, 2, 3))},
    6:  {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4, 3, 0))},
    7:  {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4, 2, 1))},
    8:  {'bits': 4, 'plan': FloatQuantizer(FloatQuantizePlan(4, 1, 2))},
    9:  {'bits': 2, 'plan': Int2BitSpecialQuantizer()},  # 最低精度
    15: {'bits': 0, 'plan': None}  # 跳过频段 (不编码)
}


# ============================================================================
# 辅助函数
# ============================================================================

def warp_steep(x, k=2):
    """
    非线性映射函数,用于将 [0,1] 区间的值进行陡峭化变换
    
    参数:
        x: 输入值 (0 到 1)
        k: 陡峭系数,越大曲线越陡
    
    返回:
        变换后的值,两端变化更快,中间更平缓
    
    用途: 在 QP 控制中,使低 QP 和高 QP 的差异更明显
    """
    return (x**k) / (x**k + (1-x)**k)


# ============================================================================
# 元数据类型定义
# ============================================================================

class QFPMetaType:
    """
    QFP 文件元数据类型定义
    
    元数据存储在文件头部,包含曲目信息、技术参数等
    格式: [Type(1B)][Length(变长)][Data(Length字节)]
    """
    
    # --- 曲目身份信息 ---
    TITLE        = 0x01  # 曲目标题
    ARTIST       = 0x02  # 演唱者/演奏者
    ALBUM_ARTIST = 0x03  # 专辑艺术家
    ALBUM        = 0x04  # 专辑名称
    
    # --- 排序与上下文 ---
    TRACK_NUMBER = 0x05  # 曲目编号 (如 "01" 或 "1/12")
    DISC_NUMBER  = 0x06  # 碟片编号 (如 "1" 或 "1/2")
    DATE         = 0x07  # 发行日期 (YYYY-MM-DD)
    GENRE        = 0x08  # 音乐流派
    COMPOSER     = 0x09  # 作曲家
    
    # --- 多媒体载荷 ---
    FRONT_COVER  = 0x10  # 封面图片 (二进制: JPG/PNG)
    LYRICS       = 0x11  # 歌词文本 (UTF-8, LRC 格式)
    
    # --- 技术信息 ---
    ENCODER_VER  = 0x20  # 编码器版本
    ENCODER_ARGS = 0x21  # 编码器参数
    COMMENT      = 0xFF  # 备注信息


# ============================================================================
# QFP3 编解码器主类
# ============================================================================

class QFP3Codec:
    """
    QFP3 音频编解码器
    
    文件格式结构:
      [Magic(4B)][Version(1B)][Header][Metadata][QuantPlan(L+R)][CompressedFrames]
    
    压缩流程:
      1. MDCT 时频变换
      2. M/S 立体声分离 (L+R, L-R)
      3. 频段分割与自适应量化级别分配
      4. 全局阈值剪枝
      5. 多比特桶分层压缩
      6. CRC 校验与帧封装
      7. Zlib 终极压缩
    
    解压流程:
      1. Zlib 解压
      2. 帧解封装与 CRC 校验
      3. 比特桶数据恢复
      4. 噪声填充 (恢复被剪枝频段的能量)
      5. iMDCT 逆变换
      6. M/S 合并为立体声
      7. 重叠相加 (Overlap-Add) 重建音频
    """
    
    def __init__(self):
        self.version = 3
        self.magic = b"QFPA"  # QFP Audio
    
    
    # ========================================================================
    # 压缩方法
    # ========================================================================
    
    def compress(self, audio_file, out_file, qp=32):
        """
        压缩音频文件为 QFP 格式
        
        参数:
            audio_file: 输入音频文件路径
            out_file: 输出 QFP 文件路径
            qp: 质量参数 (0-63), 越大质量越低、文件越小
                qp=0:  接近无损
                qp=32: 中等质量 (默认)
                qp=63: 最低质量
        
        实现细节:
            - 窗口大小: 自适应为采样率的 1/8 (向上取 2 的整数次幂)
            - 跳步大小: 窗口大小的一半 (50% 重叠)
            - 频段大小: 固定 64 系数/频段
        """
        
        # ====================================================================
        # 步骤 1: 音频预处理与参数初始化
        # ====================================================================
        
        # 读取音频文件
        data, sample_rate = sf.read(audio_file)
        
        # 计算窗口参数
        # win_size: MDCT 窗口大小,取为采样率的 1/8 并向上取 2 的整数次幂
        #           例如 44100Hz -> 8192, 48000Hz -> 8192
        # hop_size: 跳步大小,取窗口的一半 (50% 重叠)
        # band_size: 每个频段包含的 MDCT 系数数量 (固定 64)
        win_size = round_power_2(sample_rate / 8)
        band_size = 64
        hop_size = win_size // 2
        nyquist = sample_rate // 2
        num_bands = hop_size // band_size
        
        # 强制转换为立体声
        # 如果是单声道,复制为双声道
        if len(data.shape) == 1:
            data = np.column_stack((data, data))
        
        # 截取音频 (当前仅处理前 15 秒,可根据需要调整)
        # data = data[sample_rate * 0:sample_rate * 15]
        
        # 填充音频至窗口大小的整数倍
        # 这样可以确保所有音频数据都能被完整处理
        if len(data) % win_size != 0:
            padding_size = win_size - (len(data) % win_size)
            data = np.pad(data, ((0, padding_size), (0, 0)), mode='constant')
        
        # 计算总窗口数
        # 由于采用 50% 重叠,窗口数 = (样本数 / hop_size) - 1
        num_windows = (len(data) // hop_size) - 1
        
        # 创建 MDCT 窗口函数 (用于平滑边界,减少频谱泄漏)
        window = create_mdct_window(win_size)
        
        
        # ====================================================================
        # 步骤 2: 构建频率自适应 Band Plan (频段量化策略)
        # ====================================================================
        
        # 为每个频段分配量化级别和保留率
        # 策略原理:
        #   - 人耳对低频敏感,高频不敏感
        #   - 低频 (<4kHz): 使用 8-bit 高精度量化,保留率 100%
        #   - 中频 (4-12kHz): 逐渐降低精度和保留率
        #   - 高频 (>16kHz): 使用 2-bit 量化或直接跳过
        
        band_plan = []
        for i in range(0, hop_size, band_size):
            # 计算该频段的最高频率 (单位: Hz)
            max_freq = int((i + band_size) * nyquist / hop_size)
            
            # 根据频率范围确定量化级别和基础保留率
            if   max_freq < 4000:   quantizer_level, kr_base = 0,  1.0    # 低频全保留
            elif max_freq < 6000:   quantizer_level, kr_base = 3,  0.7    # 中低频
            elif max_freq < 8000:   quantizer_level, kr_base = 5,  1/2
            elif max_freq < 10000:  quantizer_level, kr_base = 6,  1/4
            elif max_freq < 12000:  quantizer_level, kr_base = 7,  1/4
            elif max_freq < 14000:  quantizer_level, kr_base = 8,  1/6
            elif max_freq < 16000:  quantizer_level, kr_base = 9,  1/6
            elif max_freq < 18000:  quantizer_level, kr_base = 9,  1/8
            elif max_freq > 20000:  quantizer_level, kr_base = 15, 0.0    # 超高频跳过
            else:                   quantizer_level, kr_base = 9,  1/10
            
            band_plan.append({
                "quantizer_level": quantizer_level,
                "kr_base": kr_base  # keep ratio base (基础保留率)
            })
        
        
        # ====================================================================
        # 步骤 3: 构造双通道静态量化方案 (L+R 和 L-R)
        # ====================================================================
        
        # QP 偏移策略:
        #   - 低 QP (高质量): offset=0, 使用默认量化级别
        #   - 高 QP (低质量): offset=3, 降低量化精度以节省空间
        offset = 3 if qp >= 48 else 0
        
        # L+R 通道方案 (Mid): 包含主要音频内容,精度较高
        lpr_plan_arr = np.array(
            [min(b['quantizer_level'] + offset, 9) for b in band_plan],
            dtype=np.uint8
        )
        
        # L-R 通道方案 (Side): 包含立体声差异,可以使用更低精度
        # 策略: 在 L+R 基础上再降 3 级 (但不超过 9)
        lmr_plan_arr = np.array(
            np.where(
                lpr_plan_arr < 9,
                np.clip(lpr_plan_arr + 3, 0, 9),
                lpr_plan_arr
            ),
            dtype=np.uint8
        )
        
        # 缓存双通道方案供后续使用
        quant_levels_cache = [lpr_plan_arr, lmr_plan_arr]
        
        
        # ====================================================================
        # 步骤 4: 写入文件头与元数据
        # ====================================================================
        
        with open(out_file, "wb") as f:
            # 4.1 写入魔数与版本号
            f.write(self.magic + struct.pack("B", self.version))
            
            # 4.2 写入核心参数
            # sample_rate: 采样率
            # win_size: MDCT 窗口大小
            # num_windows: 总窗口数
            # band_size: 每频段系数数
            f.write(struct.pack("<IIII", sample_rate, win_size, num_windows, band_size))
            
            # 4.3 写入元数据块
            # 格式: [0x55][Metadata1][Metadata2]...[0xAA]
            f.write(struct.pack('B', 0x55))  # 元数据起始标记
            
            # 示例: 写入曲目标题
            title_bytes = "QFPv3 Test Audio".encode('utf-8')
            f.write(struct.pack('B', QFPMetaType.TITLE))
            f.write(prefix_encode(len(title_bytes)))
            f.write(title_bytes)
            
            # 示例: 写入编码器版本
            version = b"QFP-Prototype-1.0"
            f.write(struct.pack('B', QFPMetaType.ENCODER_VER))
            f.write(prefix_encode(len(version)))
            f.write(version)
            
            f.write(struct.pack('B', 0xAA))  # 元数据结束标记
            
            # 4.4 写入双通道量化方案
            # 使用 4-bit 打包 (每字节存储 2 个量化级别)
            f.write(float_pack(4, lpr_plan_arr))
            f.write(float_pack(4, lmr_plan_arr))
            
            
            # ================================================================
            # 步骤 5: 核心压缩循环 - 逐帧处理
            # ================================================================
            
            io = BytesIO()  # 临时缓冲区,存储所有帧数据
            
            for i in tqdm(range(num_windows), desc="压缩进度"):
                # 5.1 提取当前窗口的左右声道数据
                L = data[i * hop_size : i * hop_size + win_size, 0]
                R = data[i * hop_size : i * hop_size + win_size, 1]
                
                # 5.2 M/S 编码 (Mid/Side)
                # LPR (L+R): 包含单声道内容 (主要能量)
                # LMR (L-R): 包含立体声宽度 (相位差异)
                # 乘以窗口函数以减少频谱泄漏
                LPR = (L + R) * window
                LMR = (L - R) * window
                
                # 5.3 计算双通道能量,用于动态调整 L-R 的压缩率
                energy_lpr = np.sum(np.abs(LPR))
                energy_lmr = np.sum(np.abs(LMR))
                
                # MS 比例: L-R 能量相对于 L+R 的比例
                # 比例越小,说明立体声信息越少,L-R 可以压缩更多
                ms_ratio = energy_lmr / (energy_lpr + 1e-6)
                
                frame_bytes = bytearray()  # 当前帧的压缩数据
                silent = True  # 静音帧标记
                
                quantize_bits = [8, 6, 4, 2]  # 比特桶列表
                
                
                # ============================================================
                # 5.4 双通道循环: 分别处理 L+R 和 L-R
                # ============================================================
                
                for ch, block in enumerate([LPR, LMR]):
                    # 5.4.1 执行 MDCT 变换
                    # 输入: 时域信号 (win_size 个样本)
                    # 输出: 频域系数 (hop_size 个系数)
                    mdct_coeff = mdct(block) / win_size
                    abs_coeff = np.abs(mdct_coeff)
                    
                    
                    # ========================================================
                    # 5.4.2 构建全局权重数组 (用于重要性评估)
                    # ========================================================
                    
                    # 权重由三个因素组成:
                    #   A. 频率权重 (f_w): 低频重要,高频次要
                    #   B. 精度权重 (p_w): 平衡不同量化级别的保留机会
                    #   C. QP 斜率 (qp_slope): 高 QP 时向低频倾斜
                    
                    weights = np.zeros(hop_size, dtype=np.float32)
                    
                    for b_idx in range(num_bands):
                        start = b_idx * band_size
                        end = start + band_size
                        q_lvl = quant_levels_cache[ch][b_idx]
                        q_bits = QUANT_LEVELS[q_lvl]['bits']
                        
                        # 因素 A: 频率权重 (来自 band_plan)
                        f_w = band_plan[b_idx]['kr_base']
                        
                        # 因素 B: 精度权重
                        # 目的: 缩小高低精度频段之间的差距
                        #       让 2-bit/4-bit 频段也有机会保留系数
                        p_w = 1 - (q_bits / 8.0) * 0.5
                        
                        # 因素 C: QP 影响
                        # QP 越大,越向低频倾斜 (保护低频基音)
                        qp_slope = 1.0 + (qp / 64.0) * (1.0 - (b_idx / num_bands))
                        
                        # 综合权重
                        weights[start:end] = f_w * p_w * qp_slope
                    
                    
                    # ========================================================
                    # 5.4.3 全局阈值剪枝
                    # ========================================================
                    
                    # 计算加权系数 (重要性分数)
                    weighted_coeffs = abs_coeff * weights
                    
                    # 计算全局保留率 (由 QP 控制)
                    # - QP=0:  保留 ~100% 系数
                    # - QP=63: 保留 ~0% 系数
                    # - L-R 通道根据 ms_ratio 动态调整
                    channel_bouns = 1 if ch == 0 else np.clip(ms_ratio * 1.5, 0.1, 0.5)
                    global_keep_ratio = np.clip(
                        max(0, warp_steep(1.0 - (qp / 63), 2)) * channel_bouns,
                        0, 0.99
                    )
                    
                    # 计算阈值: 只保留 top-N% 的重要系数
                    valid_weighted = weighted_coeffs[weights > 0]
                    if len(valid_weighted) > 0:
                        threshold = np.percentile(
                            valid_weighted,
                            (1 - global_keep_ratio) * 100
                        )
                    else:
                        threshold = 0
                    
                    
                    # ========================================================
                    # 5.4.4 逐频段量化与编码
                    # ========================================================
                    
                    bands_data = []  # 频段元数据列表
                    band_bitmap = np.zeros(num_bands, dtype=np.uint8)  # 频段激活标记
                    mdct_buckets = {8: [], 6: [], 4: [], 2: []}  # 比特桶
                    
                    for b_idx in range(num_bands):
                        start = b_idx * band_size
                        q_lvl = quant_levels_cache[ch][b_idx]
                        q_bits = QUANT_LEVELS[q_lvl]['bits']
                        
                        # 跳过级别 15 (不编码)
                        if q_lvl == 15:
                            continue
                        
                        # 提取当前频段的 MDCT 系数
                        block_data = mdct_coeff[start:start+band_size]

                        # 如果当前频段的能量为0，则跳过
                        if np.all(block_data == 0):
                            continue
                        
                        # 应用全局阈值进行剪枝
                        # 只保留加权能量 >= threshold 的系数
                        mask = weighted_coeffs[start:start+band_size] >= threshold
                        pruned_data = np.where(mask, block_data, 0)
                        
                        # 归一化: 找到最大值作为缩放因子
                        # 这样量化后的值都在 [-1, 1] 范围内
                        scale = np.max(np.abs(pruned_data))
                        if scale == 0:
                            scale = 1
                        norm_data = pruned_data / scale
                        
                        # 量化
                        quantizer = QUANT_LEVELS[q_lvl]['plan']
                        if quantizer is None:
                            continue
                        quantized = quantizer.quantize(norm_data)
                        
                        # 计算量化损失 (用于后续噪声填充)
                        # 损失 = 原始能量 - 恢复后能量
                        restored = quantizer.dequantize(quantized)
                        amp_loss = max(
                            0,
                            np.sum(np.abs(block_data / scale)) - np.sum(np.abs(restored))
                        ) * scale
                        
                        # 编码损失值为 1 字节
                        loss_byte = encode_scale(amp_loss)
                        
                        # 记录频段数据
                        band_bitmap[b_idx] = 1
                        silent = False
                        bands_data.append({
                            "scale": scale,
                            "loss_byte": loss_byte,
                            "amp_loss": amp_loss
                        })
                        
                        # 将量化后的系数放入对应的比特桶
                        mdct_buckets[q_bits].extend(quantized)
                    
                    
                    # ========================================================
                    # 5.4.5 写入通道数据
                    # ========================================================
                    
                    # 写入频段激活位图 (1 bit/频段,打包为字节)
                    frame_bytes.extend(np.packbits(band_bitmap).tobytes())
                    
                    # 写入各频段元数据 (scale + loss)
                    for b in bands_data:
                        frame_bytes.append(encode_scale(b['scale']))
                        frame_bytes.append(b['loss_byte'])
                    
                    # 写入各比特桶数据
                    for bits in quantize_bits:
                        bucket = mdct_buckets[bits]
                        
                        # 稀疏编码: 只存储非零位置和值
                        pos_info = pos_encode(bucket, min_length=3*8//bits)
                        pos_bytes = encode_pos_arr(pos_info['pos'])
                        stripped = np.array(pos_info['stripped'])
                        
                        # 对齐到字节边界
                        align = {8: 1, 6: 4, 4: 2, 2: 4, 1: 8}.get(bits, 1)
                        if len(stripped) % align != 0:
                            stripped = np.concatenate([
                                stripped,
                                np.zeros(align - (len(stripped) % align), dtype=np.uint8)
                            ])
                        
                        # 打包量化数据
                        result = float_pack(bits, stripped)
                        
                        # 写入: [位置长度][位置数据][值长度][值数据]
                        frame_bytes.extend(prefix_encode(len(pos_bytes)) + pos_bytes)
                        frame_bytes.extend(prefix_encode(len(result)) + result)
                
                
                # ============================================================
                # 5.5 封装帧并写入
                # ============================================================
                
                if silent:
                    # 静音帧: 写入空帧
                    io.write(encapsulate_frame(b''))
                else:
                    # 计算 CRC16 校验和
                    crc = crc16(bytes(frame_bytes))
                    # 封装: [CRC(2B)][帧数据]
                    io.write(encapsulate_frame(struct.pack("<H", crc) + frame_bytes))
            
            
            # ================================================================
            # 步骤 6: Zlib 最终压缩并写入文件
            # ================================================================
            
            f.write(zlib.compress(io.getvalue()))
    
    
    # ========================================================================
    # 解压方法
    # ========================================================================
    
    def decompress(self, in_file, out_file):
        """
        解压 QFP 文件为 WAV 音频
        
        参数:
            in_file: 输入 QFP 文件路径
            out_file: 输出 WAV 文件路径
        
        解压流程:
            1. 读取文件头与量化方案
            2. Zlib 解压帧数据
            3. 逐帧解析与反量化
            4. 噪声填充 (恢复被剪枝的能量)
            5. iMDCT 逆变换
            6. M/S 合并为立体声
            7. 重叠相加 (OLA) 重建波形
        """
        
        with open(in_file, "rb") as f:
            # ================================================================
            # 步骤 1: 读取文件头
            # ================================================================
            
            # 1.1 读取魔数与版本
            magic = f.read(4)
            version = struct.unpack("B", f.read(1))[0]
            
            # 1.2 读取核心参数
            sr, win_size, num_windows, band_size = struct.unpack("<IIII", f.read(16))
            
            
            # ================================================================
            # 步骤 2: 读取元数据 (如果存在)
            # ================================================================
            
            meta_flag = struct.unpack('B', f.read(1))[0]
            metadata = {}
            
            if meta_flag == 0x55:
                # 元数据块存在,逐条读取
                while True:
                    m_type = struct.unpack('B', f.read(1))[0]
                    if m_type == 0xAA:  # 结束标记
                        break
                    
                    # 读取长度与数据
                    m_len = prefix_decode(f)
                    m_data = f.read(m_len)
                    
                    metadata[m_type] = m_data
                    print(f"Found Metadata Type {m_type}, Len {m_len}, Content: {m_data}")
            else:
                # 兼容老版本: 无元数据块,回退 1 字节
                f.seek(-1, 1)
            
            
            # ================================================================
            # 步骤 3: 读取双通道量化方案
            # ================================================================
            
            hop_size = win_size // 2
            num_bands = hop_size // band_size
            plan_bytes_len = (num_bands + 1) // 2  # 4-bit 打包,每字节存 2 个值
            
            # 解包 L+R 和 L-R 的量化方案
            lpr_plan = float_unpack(4, f.read(plan_bytes_len), num_bands)
            lmr_plan = float_unpack(4, f.read(plan_bytes_len), num_bands)
            dual_ch_plans = [lpr_plan, lmr_plan]
            
            
            # ================================================================
            # 步骤 4: Zlib 解压帧数据
            # ================================================================
            
            data = zlib.decompress(f.read())
            
            # 调试: 保存解压后的原始数据
            with open("out.dump", "wb") as g:
                g.write(data)
            
            io = BytesIO(data)
            
            
            # ================================================================
            # 步骤 5: 初始化解码缓冲区
            # ================================================================
            
            window = create_mdct_window(win_size)
            recon_audio = []  # 重建的音频帧列表
            
            # OLA 缓冲区 (Overlap-Add Buffer)
            # 大小: 3 * hop_size,用于存储重叠的窗口
            out_buffer = np.zeros((2, hop_size * 3), dtype=np.float32)
            
            
            # ================================================================
            # 步骤 6: 逐帧解压循环
            # ================================================================
            
            for _ in tqdm(range(num_windows), desc="解压进度"):
                # 6.1 解封装帧
                frame_bytes = decapsulate_frame(io)
                
                # 6.2 处理静音帧
                if len(frame_bytes) == 0:
                    recon_audio.append(np.zeros((hop_size, 2)))
                    continue
                
                frame = BytesIO(frame_bytes)
                ch_recon = np.zeros((2, win_size), dtype=np.float32)
                
                # 6.3 CRC 校验
                crc_stored = struct.unpack("<H", frame.read(2))[0]
                if crc16(frame_bytes[2:]) != crc_stored:
                    raise ValueError("CRC check failed")
                
                
                # ============================================================
                # 6.4 双通道循环: 分别解码 L+R 和 L-R
                # ============================================================
                
                for i in range(2):
                    mdct_coeff = np.zeros(hop_size)
                    
                    # 6.4.1 读取频段激活位图
                    bitmap_size = int(np.ceil(num_bands / 8))
                    band_bitmap = np.unpackbits(
                        np.frombuffer(frame.read(bitmap_size), dtype=np.uint8)
                    )[:num_bands]
                    
                    
                    # ========================================================
                    # 6.4.2 读取频段元数据 (Scale + Loss)
                    # ========================================================
                    
                    band_meta = []
                    for b_idx in range(num_bands):
                        if band_bitmap[b_idx] == 1:
                            scale = decode_scale(frame.read(1)[0])
                            loss_byte = frame.read(1)[0]
                            
                            # 计算归一化损失值
                            loss_val = 0 if scale == 0 else decode_scale(loss_byte) / scale
                            band_meta.append((scale, loss_val))
                    
                    
                    # ========================================================
                    # 6.4.3 读取并恢复各比特桶数据
                    # ========================================================
                    
                    buckets_quantized = {8: [], 6: [], 4: [], 2: []}
                    
                    for bits in [8, 6, 4, 2]:
                        # 读取位置编码
                        pos_len = prefix_decode(frame)
                        pos = decode_pos_arr(BytesIO(frame.read(pos_len)))
                        
                        # 读取量化值
                        res_len = prefix_decode(frame)
                        quant_data = frame.read(res_len)
                        
                        if res_len == 0:
                            continue  # 该比特等级无数据
                        
                        # 统计该比特等级的激活频段数
                        active_bands_count = 0
                        for b_idx in range(num_bands):
                            if (band_bitmap[b_idx] == 1 and 
                                QUANT_LEVELS[dual_ch_plans[i][b_idx]]['bits'] == bits):
                                active_bands_count += 1
                        
                        # 解包量化数据
                        stripped = float_unpack(bits, quant_data, len(quant_data) * 8 // bits)
                        
                        # 恢复稀疏编码 (插入零)
                        full_bucket = pos_decode(pos, stripped, active_bands_count * band_size)
                        buckets_quantized[bits] = np.array(full_bucket, dtype=np.uint8)
                    
                    
                    # ========================================================
                    # 6.4.4 分配数据到各频段并反量化
                    # ========================================================
                    
                    meta_ptr = 0
                    bucket_ptrs = {8: 0, 6: 0, 4: 0, 2: 0}
                    amp_losses = np.zeros(num_bands)
                    
                    for b_idx in range(num_bands):
                        if band_bitmap[b_idx] == 0:
                            continue
                        
                        q_bits = QUANT_LEVELS[dual_ch_plans[i][b_idx]]['bits']
                        scale, loss_norm = band_meta[meta_ptr]
                        meta_ptr += 1
                        
                        # 从对应比特桶中提取数据
                        start_p = bucket_ptrs[q_bits]
                        end_p = start_p + band_size
                        q_slice = buckets_quantized[q_bits][start_p:end_p]
                        bucket_ptrs[q_bits] = end_p
                        
                        if len(q_slice) == 0:
                            q_slice = np.zeros(band_size, dtype=np.uint8)
                        
                        # 反量化并恢复缩放
                        quantizer = QUANT_LEVELS[dual_ch_plans[i][b_idx]]['plan']
                        mdct_coeff[b_idx*band_size : (b_idx+1)*band_size] = (
                            quantizer.dequantize(q_slice) * scale
                        )
                        
                        # 记录损失值供噪声填充使用
                        amp_losses[b_idx] = loss_norm * scale
                    
                    
                    # ========================================================
                    # 6.4.5 噪声填充 (Noise Filling)
                    # ========================================================
                    
                    # 目的: 恢复被剪枝频段的能量,避免"空洞"
                    # 方法: 在零系数位置注入与损失量相当的随机噪声
                    
                    from scipy.interpolate import interp1d
                    
                    # 构建插值点: 每个频段中心
                    x_old = np.arange(band_size // 2, hop_size, band_size)
                    
                    # 添加边界点 (防止边界突变)
                    x_points = np.concatenate([[-band_size], x_old, [hop_size + band_size]])
                    y_points = np.concatenate([[0], amp_losses, [0]])
                    
                    # 线性插值生成全频段噪声包络
                    f_interp = interp1d(x_points, y_points, kind='linear')
                    noise_envelope = f_interp(np.arange(hop_size))
                    
                    # 生成高斯白噪声
                    raw_noise = np.random.normal(0, 1.0, hop_size)
                    
                    # 逐频段归一化并注入噪声
                    for b_idx in range(num_bands):
                        idx_range = slice(b_idx * band_size, (b_idx + 1) * band_size)
                        curr_noise = raw_noise[idx_range]
                        curr_sum = np.sum(np.abs(curr_noise))
                        
                        if curr_sum > 0:
                            # 只在零系数位置注入噪声
                            zero_mask = (mdct_coeff[idx_range] == 0)
                            target_amp = noise_envelope[idx_range]
                            mdct_coeff[idx_range] += (
                                (curr_noise / curr_sum) * target_amp * zero_mask
                            )
                    
                    
                    # ========================================================
                    # 6.4.6 iMDCT 逆变换
                    # ========================================================
                    
                    ch_recon[i] = imdct(mdct_coeff * win_size) * window
                
                
                # ============================================================
                # 6.5 M/S 解码: 合并为立体声
                # ============================================================
                
                LPR, LMR = ch_recon
                
                # 累加到 OLA 缓冲区 (重叠相加)
                out_buffer[0, hop_size:] += LPR
                out_buffer[1, hop_size:] += LMR
                
                # 从 L+R 和 L-R 恢复左右声道
                L_out = (out_buffer[0, :hop_size] + out_buffer[1, :hop_size]) / 2
                R_out = (out_buffer[0, :hop_size] - out_buffer[1, :hop_size]) / 2
                
                # 保存当前帧
                recon_audio.append(np.column_stack((L_out, R_out)))
                
                # 缓冲区平移 (移除已输出部分)
                out_buffer = np.roll(out_buffer, -hop_size, axis=1)
                out_buffer[:, -hop_size:] = 0
            
            
            # ================================================================
            # 步骤 7: 合并音频并写入文件
            # ================================================================
            
            sf.write(out_file, np.concatenate(recon_audio, axis=0), sr)


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    encoder = QFP3Codec()

    import sys
    
    # 压缩示例
    encoder.compress(
        sys.argv[1],
        "test.qfp",
        qp=41
    )
    
    # 解压示例
    encoder.decompress("test.qfp", "result.wav")