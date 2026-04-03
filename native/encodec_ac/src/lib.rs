use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyEOFError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyBytes;

const EPS_EDGE: f64 = 9.094947017729282e-13;
const EPS_PERTURB: f64 = 8.673617379884035e-19;

fn require_torch_tensor_layout(
    tensor: &Bound<'_, PyAny>,
    expected_dtype: &str,
    expected_dim: usize,
) -> PyResult<Vec<usize>> {
    let is_contiguous = tensor.call_method0("is_contiguous")?.extract::<bool>()?;
    if !is_contiguous {
        return Err(PyValueError::new_err("tensor must be contiguous"));
    }

    let device = tensor.getattr("device")?.getattr("type")?.extract::<String>()?;
    if device != "cpu" {
        return Err(PyValueError::new_err("tensor must be on CPU"));
    }

    let dtype = tensor.getattr("dtype")?.str()?.to_str()?.to_owned();
    if dtype != expected_dtype {
        return Err(PyValueError::new_err(format!(
            "tensor must have dtype {expected_dtype}, got {dtype}"
        )));
    }

    let shape = tensor.getattr("shape")?.extract::<Vec<usize>>()?;
    if shape.len() != expected_dim {
        return Err(PyValueError::new_err(format!(
            "tensor must be {expected_dim}D, got {}D",
            shape.len()
        )));
    }
    Ok(shape)
}

fn torch_f64_tensor_2d<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<(usize, usize, &'py [f64])> {
    let shape = require_torch_tensor_layout(tensor, "torch.float64", 2)?;
    let n_bins = shape[0];
    let n_cols = shape[1];
    let ptr = tensor.call_method0("data_ptr")?.extract::<usize>()?;
    let len = n_bins
        .checked_mul(n_cols)
        .ok_or_else(|| PyValueError::new_err("tensor shape is too large"))?;
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const f64, len) };
    Ok((n_bins, n_cols, slice))
}

fn torch_i64_tensor_1d<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<(usize, &'py [i64])> {
    let shape = require_torch_tensor_layout(tensor, "torch.int64", 1)?;
    let len = shape[0];
    let ptr = tensor.call_method0("data_ptr")?.extract::<usize>()?;
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const i64, len) };
    Ok((len, slice))
}

fn torch_i64_tensor_1d_mut<'py>(
    tensor: &Bound<'py, PyAny>,
) -> PyResult<(usize, &'py mut [i64])> {
    let shape = require_torch_tensor_layout(tensor, "torch.int64", 1)?;
    let len = shape[0];
    let ptr = tensor.call_method0("data_ptr")?.extract::<usize>()?;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i64, len) };
    Ok((len, slice))
}

fn counts_from_pdf_flat(pdf: &[f64], fp_scale: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(pdf.len());
    let scale = fp_scale as f64;
    for (idx, value) in pdf.iter().enumerate() {
        let mut x = value.max(0.0) * scale;
        let frac = x - x.floor();
        if frac <= EPS_EDGE || frac >= 1.0 - EPS_EDGE {
            let sign = if idx % 2 == 0 { -1.0 } else { 1.0 };
            x = (x + sign * EPS_PERTURB).max(0.0);
        }
        out.push(x.floor() as i64);
    }
    out
}

fn deterministic_cdf_multi_impl(
    pdf: &[f64],
    n_bins: usize,
    n_cols: usize,
    total_range_bits: u32,
    fp_scale: i64,
    min_range: i64,
) -> PyResult<Vec<i64>> {
    if n_bins == 0 || n_cols == 0 {
        return Err(PyValueError::new_err("pdf_mat must be non-empty"));
    }
    if pdf.len() != n_bins * n_cols {
        return Err(PyValueError::new_err("pdf_mat shape does not match buffer length"));
    }

    let total = 1_i64
        .checked_shl(total_range_bits)
        .ok_or_else(|| PyValueError::new_err("total_range_bits too large"))?;
    let alloc = total - min_range * (n_bins as i64);
    if alloc <= 0 {
        return Err(PyValueError::new_err("invalid total_range_bits/min_range combination"));
    }

    let mut normalized = vec![0.0_f64; pdf.len()];
    for col in 0..n_cols {
        let mut sum = 0.0_f64;
        for row in 0..n_bins {
            let v = pdf[row * n_cols + col].max(0.0);
            normalized[row * n_cols + col] = v;
            sum += v;
        }
        if !sum.is_finite() || sum <= 0.0 {
            for row in 0..n_bins {
                normalized[row * n_cols + col] = 1.0;
            }
        }
    }

    let mut counts = counts_from_pdf_flat(&normalized, fp_scale);
    for col in 0..n_cols {
        let mut sum = 0_i64;
        for row in 0..n_bins {
            sum += counts[row * n_cols + col];
        }
        if sum <= 0 {
            for row in 0..n_bins {
                counts[row * n_cols + col] = 1;
            }
        }
    }

    let mut cdf = vec![0_i64; pdf.len()];
    for col in 0..n_cols {
        let mut num_sum = 0_i64;
        for row in 0..n_bins {
            num_sum += counts[row * n_cols + col];
        }
        if num_sum <= 0 {
            return Err(PyValueError::new_err("invalid zero-count column"));
        }

        let mut base = vec![0_i64; n_bins];
        let mut base_sum = 0_i64;
        for row in 0..n_bins {
            let num = counts[row * n_cols + col];
            let value = (alloc * num) / num_sum;
            base[row] = value;
            base_sum += value;
        }
        let remainder = alloc - base_sum;
        if remainder > 0 {
            let mut order: Vec<(i64, usize)> = (0..n_bins)
                .map(|row| {
                    let num = counts[row * n_cols + col];
                    let prio = (alloc * num) - (num_sum * base[row]);
                    let key = prio * ((n_bins as i64) + 1) - (row as i64);
                    (key, row)
                })
                .collect();
            order.sort_by(|a, b| b.cmp(a));
            for (_, row) in order.into_iter().take(remainder as usize) {
                base[row] += 1;
            }
        }

        let mut running = 0_i64;
        for row in 0..n_bins {
            running += base[row] + min_range;
            cdf[row * n_cols + col] = running;
        }
        if running != total {
            return Err(PyValueError::new_err("cdf sum mismatch"));
        }
    }
    Ok(cdf)
}

struct BitWriter {
    current_value: u64,
    current_bits: u8,
    bytes: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            current_value: 0,
            current_bits: 0,
            bytes: Vec::new(),
        }
    }

    fn push_bit(&mut self, bit: u8) {
        self.current_value += (bit as u64) << self.current_bits;
        self.current_bits += 1;
        while self.current_bits >= 8 {
            let lower = (self.current_value & 0xff) as u8;
            self.current_bits -= 8;
            self.current_value >>= 8;
            self.bytes.push(lower);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.current_bits > 0 {
            self.bytes.push(self.current_value as u8);
            self.current_value = 0;
            self.current_bits = 0;
        }
        self.bytes
    }
}

struct BitReader {
    data: Vec<u8>,
    offset: usize,
    current_value: u64,
    current_bits: u8,
}

impl BitReader {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            offset: 0,
            current_value: 0,
            current_bits: 0,
        }
    }

    fn pull_bit(&mut self) -> Option<u8> {
        while self.current_bits < 1 {
            let byte = *self.data.get(self.offset)?;
            self.offset += 1;
            self.current_value += (byte as u64) << self.current_bits;
            self.current_bits += 8;
        }
        let out = (self.current_value & 1) as u8;
        self.current_value >>= 1;
        self.current_bits -= 1;
        Some(out)
    }
}

#[pyclass]
struct ArithmeticEncoder {
    total_range_bits: u32,
    low: u64,
    high: u64,
    max_bit: i32,
    writer: BitWriter,
}

#[pymethods]
impl ArithmeticEncoder {
    #[new]
    #[pyo3(signature = (total_range_bits = 24))]
    fn new(total_range_bits: u32) -> PyResult<Self> {
        if total_range_bits > 30 {
            return Err(PyValueError::new_err("total_range_bits must be <= 30"));
        }
        Ok(Self {
            total_range_bits,
            low: 0,
            high: 0,
            max_bit: -1,
            writer: BitWriter::new(),
        })
    }

    fn push_pdf_symbols(
        &mut self,
        pdf_mat: PyReadonlyArray2<f64>,
        symbols: Vec<usize>,
        fp_scale: i64,
        min_range: i64,
    ) -> PyResult<()> {
        let shape = pdf_mat.shape();
        let n_bins = shape[0];
        let n_cols = shape[1];
        if symbols.len() != n_cols {
            return Err(PyValueError::new_err("symbols length must match the pdf column count"));
        }
        let pdf = pdf_mat
            .as_slice()
            .map_err(|_| PyValueError::new_err("pdf_mat must be C-contiguous"))?;
        let cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
        )?;
        for (col, symbol) in symbols.into_iter().enumerate() {
            self.push_symbol(symbol, &cdf, n_bins, n_cols, col)?;
        }
        Ok(())
    }

    fn push_pdf_symbols_torch(
        &mut self,
        pdf_mat: &Bound<'_, PyAny>,
        symbols: &Bound<'_, PyAny>,
        fp_scale: i64,
        min_range: i64,
    ) -> PyResult<()> {
        let (n_bins, n_cols, pdf) = torch_f64_tensor_2d(pdf_mat)?;
        let (symbol_len, symbol_slice) = torch_i64_tensor_1d(symbols)?;
        if symbol_len != n_cols {
            return Err(PyValueError::new_err("symbols length must match the pdf column count"));
        }
        let cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
        )?;
        for (col, symbol) in symbol_slice.iter().enumerate() {
            if *symbol < 0 {
                return Err(PyValueError::new_err("symbols must be non-negative"));
            }
            self.push_symbol(*symbol as usize, &cdf, n_bins, n_cols, col)?;
        }
        Ok(())
    }

    fn finish<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        while self.max_bit >= 0 {
            let bit = ((self.low >> (self.max_bit as u32)) & 1) as u8;
            self.writer.push_bit(bit);
            self.max_bit -= 1;
        }
        let bytes = std::mem::replace(&mut self.writer, BitWriter::new()).finish();
        PyBytes::new(py, &bytes)
    }
}

impl ArithmeticEncoder {
    fn delta(&self) -> u64 {
        self.high - self.low + 1
    }

    fn flush_common_prefix(&mut self) {
        while self.max_bit >= 0 {
            let b1 = self.low >> (self.max_bit as u32);
            let b2 = self.high >> (self.max_bit as u32);
            if b1 == b2 {
                self.low -= b1 << (self.max_bit as u32);
                self.high -= b1 << (self.max_bit as u32);
                self.max_bit -= 1;
                self.writer.push_bit(b1 as u8);
            } else {
                break;
            }
        }
    }

    fn push_symbol(
        &mut self,
        symbol: usize,
        cdf: &[i64],
        n_bins: usize,
        n_cols: usize,
        col: usize,
    ) -> PyResult<()> {
        while self.delta() < (1_u64 << self.total_range_bits) {
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.max_bit += 1;
        }
        if symbol >= n_bins {
            return Err(PyValueError::new_err("symbol out of range"));
        }
        let total = 1_u64 << self.total_range_bits;
        let rng = self.delta();
        let cum_high = cdf[symbol * n_cols + col] as u64;
        let cum_low = if symbol == 0 {
            0
        } else {
            cdf[(symbol - 1) * n_cols + col] as u64
        };
        let base = self.low;
        self.low = base + (rng * cum_low) / total;
        self.high = base + (rng * cum_high) / total - 1;
        self.flush_common_prefix();
        Ok(())
    }
}

#[pyclass]
struct ArithmeticDecoder {
    total_range_bits: u32,
    low: u64,
    high: u64,
    current: u64,
    max_bit: i32,
    reader: BitReader,
}

#[pymethods]
impl ArithmeticDecoder {
    #[new]
    #[pyo3(signature = (data, total_range_bits = 24))]
    fn new(data: &Bound<'_, PyBytes>, total_range_bits: u32) -> PyResult<Self> {
        if total_range_bits > 30 {
            return Err(PyValueError::new_err("total_range_bits must be <= 30"));
        }
        Ok(Self {
            total_range_bits,
            low: 0,
            high: 0,
            current: 0,
            max_bit: -1,
            reader: BitReader::new(data.as_bytes().to_vec()),
        })
    }

    fn pull_symbols(
        &mut self,
        pdf_mat: PyReadonlyArray2<f64>,
        fp_scale: i64,
        min_range: i64,
    ) -> PyResult<Vec<usize>> {
        let shape = pdf_mat.shape();
        let n_bins = shape[0];
        let n_cols = shape[1];
        let pdf = pdf_mat
            .as_slice()
            .map_err(|_| PyValueError::new_err("pdf_mat must be C-contiguous"))?;
        let cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
        )?;
        let mut out = Vec::with_capacity(n_cols);
        for col in 0..n_cols {
            let symbol = self.pull_symbol(&cdf, n_bins, n_cols, col)?;
            out.push(symbol);
        }
        Ok(out)
    }

    fn pull_symbols_into_torch(
        &mut self,
        pdf_mat: &Bound<'_, PyAny>,
        out_symbols: &Bound<'_, PyAny>,
        fp_scale: i64,
        min_range: i64,
    ) -> PyResult<()> {
        let (n_bins, n_cols, pdf) = torch_f64_tensor_2d(pdf_mat)?;
        let (out_len, out_slice) = torch_i64_tensor_1d_mut(out_symbols)?;
        if out_len != n_cols {
            return Err(PyValueError::new_err(
                "output tensor length must match the pdf column count",
            ));
        }
        let cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
        )?;
        for col in 0..n_cols {
            let symbol = self.pull_symbol(&cdf, n_bins, n_cols, col)?;
            out_slice[col] = symbol as i64;
        }
        Ok(())
    }
}

impl ArithmeticDecoder {
    fn delta(&self) -> u64 {
        self.high - self.low + 1
    }

    fn flush_common_prefix(&mut self) {
        while self.max_bit >= 0 {
            let b1 = self.low >> (self.max_bit as u32);
            let b2 = self.high >> (self.max_bit as u32);
            if b1 == b2 {
                self.low -= b1 << (self.max_bit as u32);
                self.high -= b1 << (self.max_bit as u32);
                self.current -= b1 << (self.max_bit as u32);
                self.max_bit -= 1;
            } else {
                break;
            }
        }
    }

    fn pull_symbol(
        &mut self,
        cdf: &[i64],
        n_bins: usize,
        n_cols: usize,
        col: usize,
    ) -> PyResult<usize> {
        while self.delta() < (1_u64 << self.total_range_bits) {
            let bit = self
                .reader
                .pull_bit()
                .ok_or_else(|| PyEOFError::new_err("stream exhausted"))? as u64;
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.current = (self.current << 1) | bit;
            self.max_bit += 1;
        }

        let total = 1_u64 << self.total_range_bits;
        let rng = self.delta();
        let target = (((self.current - self.low + 1) * total) - 1) / rng;
        let mut lo = 0usize;
        let mut hi = n_bins;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let value = cdf[mid * n_cols + col] as u64;
            if target < value {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if lo >= n_bins {
            return Err(PyValueError::new_err("binary search failed"));
        }
        let symbol = lo;
        let cum_high = cdf[symbol * n_cols + col] as u64;
        let cum_low = if symbol == 0 {
            0
        } else {
            cdf[(symbol - 1) * n_cols + col] as u64
        };
        let base = self.low;
        self.low = base + (rng * cum_low) / total;
        self.high = base + (rng * cum_high) / total - 1;
        self.flush_common_prefix();
        Ok(symbol)
    }
}

#[pyfunction]
fn deterministic_cdf_multi<'py>(
    py: Python<'py>,
    pdf_mat: PyReadonlyArray2<f64>,
    total_range_bits: u32,
    fp_scale: i64,
    min_range: i64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let shape = pdf_mat.shape();
    let n_bins = shape[0];
    let n_cols = shape[1];
    let pdf = pdf_mat
        .as_slice()
        .map_err(|_| PyValueError::new_err("pdf_mat must be C-contiguous"))?;
    let cdf = deterministic_cdf_multi_impl(pdf, n_bins, n_cols, total_range_bits, fp_scale, min_range)?;
    let rows: Vec<Vec<i64>> = (0..n_bins)
        .map(|row| {
            (0..n_cols)
                .map(|col| cdf[row * n_cols + col])
                .collect::<Vec<_>>()
        })
        .collect();
    Ok(PyArray2::from_vec2(py, &rows)?)
}

#[pymodule]
fn encodec_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArithmeticEncoder>()?;
    m.add_class::<ArithmeticDecoder>()?;
    m.add_function(wrap_pyfunction!(deterministic_cdf_multi, m)?)?;
    Ok(())
}
