use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" ASCII as little-endian u32

/// GGUF tensor data types we support dequantizing
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            30 => Ok(Self::BF16),
            other => bail!("Unsupported GGML type: {}", other),
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::BF16 => "BF16",
            _ => "unknown",
        }
    }

    /// Bytes per element (for non-quantized types)
    fn element_size(&self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 => Some(2),
            Self::F64 => Some(8),
            Self::BF16 => Some(2),
            Self::I8 => Some(1),
            Self::I16 => Some(2),
            Self::I32 => Some(4),
            Self::I64 => Some(8),
            _ => None, // quantized types use block sizes
        }
    }

    /// Block size for quantized types (number of elements per block)
    fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            _ => 1,
        }
    }

    /// Bytes per block for quantized types
    fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 2 + 32 / 2,    // scale(f16) + 32 nibbles = 18
            Self::Q4_1 => 2 + 2 + 32 / 2, // scale + min + nibbles = 20
            Self::Q5_0 => 2 + 4 + 32 / 2, // scale + high bits + nibbles = 22
            Self::Q5_1 => 2 + 2 + 4 + 32 / 2, // = 24
            Self::Q8_0 => 2 + 32,          // scale + 32 bytes = 34
            Self::Q8_1 => 4 + 4 + 32,     // scale(f32) + sum(f32) + 32 bytes = 40
            // K-quant block sizes (256 elements per block)
            Self::Q2K => 256 / 16 + 256 / 4 + 2 + 2, // 84 bytes
            Self::Q3K => 256 / 8 + 256 / 4 + 12 + 2, // 110 bytes
            Self::Q4K => 2 + 2 + 12 + 256 / 2,        // 144 bytes
            Self::Q5K => 2 + 2 + 12 + 256 / 2 + 32,   // 176 bytes
            Self::Q6K => 256 / 2 + 256 / 4 + 256 / 16 + 2, // 210 bytes
            Self::Q8K => 4 + 256 + 16 * 2,             // 292 bytes (f32 scale + 256 int8 + 16 f16 sums)
            _ => 0,
        }
    }
}

/// Metadata value types in GGUF
#[derive(Debug, Clone)]
pub enum MetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<MetaValue>),
}

impl std::fmt::Display for MetaValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaValue::Uint8(v) => write!(f, "{}", v),
            MetaValue::Int8(v) => write!(f, "{}", v),
            MetaValue::Uint16(v) => write!(f, "{}", v),
            MetaValue::Int16(v) => write!(f, "{}", v),
            MetaValue::Uint32(v) => write!(f, "{}", v),
            MetaValue::Int32(v) => write!(f, "{}", v),
            MetaValue::Float32(v) => write!(f, "{}", v),
            MetaValue::Bool(v) => write!(f, "{}", v),
            MetaValue::String(v) => write!(f, "\"{}\"", v),
            MetaValue::Uint64(v) => write!(f, "{}", v),
            MetaValue::Int64(v) => write!(f, "{}", v),
            MetaValue::Float64(v) => write!(f, "{}", v),
            MetaValue::Array(v) => write!(f, "[{} elements]", v.len()),
        }
    }
}

/// A single tensor's metadata (not the data itself)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64, // offset from start of tensor data section
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    pub fn shape_str(&self) -> String {
        if self.dims.is_empty() {
            "scalar".to_string()
        } else {
            format!("[{}]", self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
        }
    }

    /// Total bytes this tensor occupies in the file
    pub fn data_size(&self) -> usize {
        let n = self.n_elements() as usize;
        if let Some(elem_sz) = self.dtype.element_size() {
            n * elem_sz
        } else {
            let bs = self.dtype.block_size();
            let bb = self.dtype.block_bytes();
            let n_blocks = (n + bs - 1) / bs;
            n_blocks * bb
        }
    }
}

/// Parsed GGUF file (memory-mapped)
pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: usize, // byte offset where tensor data begins
}

/// Cursor for reading from the mmap
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!("Unexpected EOF at offset {} (need {} bytes, have {})",
                  self.pos, n, self.remaining());
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        if len > 1_000_000 {
            bail!("Implausible string length: {}", len);
        }
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).context("Invalid UTF-8 in GGUF string")
    }

    fn read_meta_value(&mut self, vtype: u32) -> Result<MetaValue> {
        match vtype {
            0 => Ok(MetaValue::Uint8(self.read_u8()?)),
            1 => Ok(MetaValue::Int8(self.read_i8()?)),
            2 => Ok(MetaValue::Uint16(self.read_u16()?)),
            3 => Ok(MetaValue::Int16(self.read_i16()?)),
            4 => Ok(MetaValue::Uint32(self.read_u32()?)),
            5 => Ok(MetaValue::Int32(self.read_i32()?)),
            6 => Ok(MetaValue::Float32(self.read_f32()?)),
            7 => Ok(MetaValue::Bool(self.read_bool()?)),
            8 => Ok(MetaValue::String(self.read_string()?)),
            9 => {
                // Array: element_type (u32), count (u64), then elements
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                if count > 10_000_000 {
                    bail!("Implausible array length: {}", count);
                }
                let mut arr = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    arr.push(self.read_meta_value(elem_type)?);
                }
                Ok(MetaValue::Array(arr))
            }
            10 => Ok(MetaValue::Uint64(self.read_u64()?)),
            11 => Ok(MetaValue::Int64(self.read_i64()?)),
            12 => Ok(MetaValue::Float64(self.read_f64()?)),
            other => bail!("Unknown GGUF metadata value type: {}", other),
        }
    }
}

impl GgufFile {
    /// Open and parse a GGUF file using memory-mapped IO
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;

        // Safety: we only read from the mmap, never write
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap {}", path.display()))?;

        if mmap.len() < 24 {
            bail!("File too small to be GGUF ({}B)", mmap.len());
        }

        let mut cur = Cursor::new(&mmap);

        // Header
        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            bail!("Not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})", magic, GGUF_MAGIC);
        }

        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            bail!("Unsupported GGUF version: {} (supported: 2, 3)", version);
        }

        let tensor_count = cur.read_u64()? as usize;
        let metadata_kv_count = cur.read_u64()? as usize;

        if tensor_count > 100_000 || metadata_kv_count > 100_000 {
            bail!("Implausible counts: tensors={}, metadata={}", tensor_count, metadata_kv_count);
        }

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let vtype = cur.read_u32()?;
            let value = cur.read_meta_value(vtype)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            if n_dims > 8 {
                bail!("Implausible tensor ndims: {}", n_dims);
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            let dtype = GgmlType::from_u32(cur.read_u32()?)?;
            let offset = cur.read_u64()?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }

        // Tensor data starts at next alignment boundary (32 bytes) after header+metadata+tensor_infos
        let alignment = match metadata.get("general.alignment") {
            Some(MetaValue::Uint32(a)) => *a as usize,
            Some(MetaValue::Uint64(a)) => *a as usize,
            _ => 32,
        };
        let tensor_data_offset = (cur.pos + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            mmap,
            version,
            metadata,
            tensors,
            tensor_data_offset,
        })
    }

    /// Get raw bytes for a tensor
    fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = self.tensor_data_offset + info.offset as usize;
        let size = info.data_size();
        let end = start + size;
        if end > self.mmap.len() {
            bail!("Tensor '{}' extends past EOF (need {}..{}, file is {}B)",
                  info.name, start, end, self.mmap.len());
        }
        Ok(&self.mmap[start..end])
    }

    /// Find a tensor by name
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Extract a tensor as f32 values. Supports F32 and F16 dequantization.
    pub fn extract_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.find_tensor(name)
            .with_context(|| format!("Tensor '{}' not found", name))?;
        let bytes = self.tensor_bytes(info)?;
        let n = info.n_elements() as usize;

        match info.dtype {
            GgmlType::F32 => {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let off = i * 4;
                    out[i] = f32::from_le_bytes([
                        bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
                    ]);
                }
                Ok(out)
            }
            GgmlType::F16 => {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    out[i] = f16_to_f32(bits);
                }
                Ok(out)
            }
            GgmlType::BF16 => {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    out[i] = bf16_to_f32(bits);
                }
                Ok(out)
            }
            GgmlType::Q8_0 => {
                dequant_q8_0(bytes, n)
            }
            GgmlType::Q4_0 => {
                dequant_q4_0(bytes, n)
            }
            other => {
                bail!("Dequantization not implemented for type {:?} — only F32, F16, BF16, Q8_0, Q4_0 supported", other);
            }
        }
    }

    /// Print a summary of all tensors
    pub fn print_layers(&self) {
        println!("GGUF v{} — {} tensors, {} metadata keys",
                 self.version, self.tensors.len(), self.metadata.len());
        println!();

        // Print key metadata
        let interesting = [
            "general.architecture",
            "general.name",
            "general.quantization_version",
            "general.file_type",
        ];
        for key in &interesting {
            if let Some(val) = self.metadata.get(*key) {
                println!("  {}: {}", key, val);
            }
        }
        println!();

        // Print tensors
        println!("{:<60} {:>6}  {:<30}  {:>12}",
                 "TENSOR", "TYPE", "SHAPE", "SIZE");
        println!("{}", "-".repeat(115));

        let mut total_bytes: u64 = 0;
        for t in &self.tensors {
            let size = t.data_size();
            total_bytes += size as u64;
            let size_str = if size >= 1_048_576 {
                format!("{:.1} MB", size as f64 / 1_048_576.0)
            } else if size >= 1024 {
                format!("{:.1} KB", size as f64 / 1024.0)
            } else {
                format!("{} B", size)
            };
            println!("{:<60} {:>6}  {:<30}  {:>12}",
                     t.name, t.dtype.name(), t.shape_str(), size_str);
        }

        println!("{}", "-".repeat(115));
        let total_str = if total_bytes >= 1_073_741_824 {
            format!("{:.2} GB", total_bytes as f64 / 1_073_741_824.0)
        } else {
            format!("{:.1} MB", total_bytes as f64 / 1_048_576.0)
        };
        println!("Total tensor data: {}", total_str);
    }

    /// Get raw Q4_0 nibble values (0..15) for a tensor. Returns per-block nibbles.
    /// Each inner array has 32 nibbles (the block size).
    pub fn extract_q4_0_nibbles(&self, name: &str) -> Result<Vec<[u8; 32]>> {
        let info = self.find_tensor(name)
            .with_context(|| format!("Tensor '{}' not found", name))?;
        if info.dtype != GgmlType::Q4_0 {
            bail!("Tensor '{}' is {:?}, not Q4_0", name, info.dtype);
        }
        let bytes = self.tensor_bytes(info)?;
        let n = info.n_elements() as usize;
        let block_bytes = 18usize; // 2 (f16 scale) + 16 (nibble bytes)
        let n_blocks = (n + 31) / 32;
        let mut result = Vec::with_capacity(n_blocks);

        for bi in 0..n_blocks {
            let boff = bi * block_bytes;
            if boff + block_bytes > bytes.len() {
                break;
            }
            let mut nibbles = [0u8; 32];
            for j in 0..16 {
                let byte = bytes[boff + 2 + j];
                nibbles[j] = byte & 0x0F;        // lo nibble
                nibbles[j + 16] = (byte >> 4) & 0x0F; // hi nibble
            }
            result.push(nibbles);
        }

        Ok(result)
    }
}

/// Convert IEEE 754 half-precision (f16) to f32
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = 1u32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 - e + 1) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf/NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normal
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Convert bfloat16 to f32 (just shift left by 16)
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Dequantize Q8_0 blocks: each block = 1 f16 scale + 32 int8 values
fn dequant_q8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 2 + 32; // f16 scale + 32 int8s
    let n_blocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for bi in 0..n_blocks {
        let boff = bi * block_bytes;
        if boff + block_bytes > bytes.len() {
            bail!("Q8_0 block {} extends past data", bi);
        }
        let scale = f16_to_f32(u16::from_le_bytes([bytes[boff], bytes[boff + 1]]));
        let base_idx = bi * block_size;
        for j in 0..block_size {
            let idx = base_idx + j;
            if idx >= n {
                break;
            }
            let q = bytes[boff + 2 + j] as i8;
            out[idx] = q as f32 * scale;
        }
    }
    Ok(out)
}

/// Dequantize Q4_0 blocks: each block = 1 f16 scale + 16 bytes (32 nibbles)
fn dequant_q4_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 2 + 16; // f16 scale + 16 bytes (2 nibbles each)
    let n_blocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for bi in 0..n_blocks {
        let boff = bi * block_bytes;
        if boff + block_bytes > bytes.len() {
            bail!("Q4_0 block {} extends past data", bi);
        }
        let scale = f16_to_f32(u16::from_le_bytes([bytes[boff], bytes[boff + 1]]));
        let base_idx = bi * block_size;
        for j in 0..16 {
            let byte = bytes[boff + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            let idx_lo = base_idx + j;
            let idx_hi = base_idx + j + 16;
            if idx_lo < n {
                out[idx_lo] = lo as f32 * scale;
            }
            if idx_hi < n {
                out[idx_hi] = hi as f32 * scale;
            }
        }
    }
    Ok(out)
}

/// Create a minimal synthetic GGUF v3 file for testing
pub fn create_test_gguf(path: &Path) -> Result<()> {
    use std::io::Write;

    let mut buf: Vec<u8> = Vec::new();

    // Magic
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count: 2
    buf.extend_from_slice(&2u64.to_le_bytes());
    // Metadata KV count: 2
    buf.extend_from_slice(&2u64.to_le_bytes());

    // Metadata: general.architecture = "test"
    write_gguf_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes()); // type = string
    write_gguf_string(&mut buf, "test");

    // Metadata: general.name = "nd-test"
    write_gguf_string(&mut buf, "general.name");
    buf.extend_from_slice(&8u32.to_le_bytes()); // type = string
    write_gguf_string(&mut buf, "nd-test");

    // Tensor info 1: "weight.0" shape [4, 3] F32
    write_gguf_string(&mut buf, "weight.0");
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    buf.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
    buf.extend_from_slice(&3u64.to_le_bytes()); // dim[1]
    buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

    // Tensor info 2: "bias.0" shape [4] F32
    let t1_size: u64 = 4 * 3 * 4; // 12 floats * 4 bytes
    write_gguf_string(&mut buf, "bias.0");
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
    buf.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
    buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
    buf.extend_from_slice(&t1_size.to_le_bytes()); // offset after first tensor

    // Pad to 32-byte alignment
    let alignment = 32;
    let pad_needed = (alignment - (buf.len() % alignment)) % alignment;
    buf.extend(vec![0u8; pad_needed]);

    // Tensor data 1: weight.0 = 12 f32 values (1.0..12.0)
    for i in 1..=12u32 {
        buf.extend_from_slice(&(i as f32).to_le_bytes());
    }

    // Tensor data 2: bias.0 = 4 f32 values (0.1, 0.2, 0.3, 0.4)
    for i in 1..=4u32 {
        buf.extend_from_slice(&(i as f32 * 0.1).to_le_bytes());
    }

    let mut file = File::create(path)
        .with_context(|| format!("Failed to create {}", path.display()))?;
    file.write_all(&buf)?;
    Ok(())
}

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}
