#include <torch/extension.h>

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kEpsEdge = 9.094947017729282e-13;
constexpr double kEpsPerturb = 8.673617379884035e-19;

void check_pdf_mat(const torch::Tensor& pdf_mat) {
    TORCH_CHECK(pdf_mat.device().is_cpu(), "pdf_mat must be on CPU");
    TORCH_CHECK(pdf_mat.scalar_type() == torch::kFloat64, "pdf_mat must have dtype torch.float64");
    TORCH_CHECK(pdf_mat.dim() == 2, "pdf_mat must be 2D");
    TORCH_CHECK(pdf_mat.is_contiguous(), "pdf_mat must be contiguous");
}

void check_symbol_tensor(const torch::Tensor& symbols, int64_t expected_len, const char* name) {
    TORCH_CHECK(symbols.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(symbols.scalar_type() == torch::kLong, name, " must have dtype torch.int64");
    TORCH_CHECK(symbols.dim() == 1, name, " must be 1D");
    TORCH_CHECK(symbols.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(symbols.numel() == expected_len, name, " length must match the pdf column count");
}

std::vector<int64_t> counts_from_pdf_flat(const double* pdf, int64_t len, int64_t fp_scale) {
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(len));
    const double scale = static_cast<double>(fp_scale);
    for (int64_t idx = 0; idx < len; ++idx) {
        double x = std::max(pdf[idx], 0.0) * scale;
        const double frac = x - std::floor(x);
        if (frac <= kEpsEdge || frac >= 1.0 - kEpsEdge) {
            const double sign = (idx % 2 == 0) ? -1.0 : 1.0;
            x = std::max(x + sign * kEpsPerturb, 0.0);
        }
        out.push_back(static_cast<int64_t>(std::floor(x)));
    }
    return out;
}

std::vector<int64_t> deterministic_cdf_multi_impl(
    const double* pdf,
    int64_t n_bins,
    int64_t n_cols,
    int64_t total_range_bits,
    int64_t fp_scale,
    int64_t min_range
) {
    TORCH_CHECK(n_bins > 0 && n_cols > 0, "pdf_mat must be non-empty");
    TORCH_CHECK(total_range_bits >= 0 && total_range_bits <= 30, "total_range_bits must be between 0 and 30");

    const int64_t total = int64_t{1} << total_range_bits;
    const int64_t alloc = total - min_range * n_bins;
    TORCH_CHECK(alloc > 0, "invalid total_range_bits/min_range combination");

    const int64_t len = n_bins * n_cols;
    std::vector<double> normalized(static_cast<size_t>(len), 0.0);
    for (int64_t col = 0; col < n_cols; ++col) {
        double sum = 0.0;
        for (int64_t row = 0; row < n_bins; ++row) {
            const double value = std::max(pdf[row * n_cols + col], 0.0);
            normalized[static_cast<size_t>(row * n_cols + col)] = value;
            sum += value;
        }
        if (!std::isfinite(sum) || sum <= 0.0) {
            for (int64_t row = 0; row < n_bins; ++row) {
                normalized[static_cast<size_t>(row * n_cols + col)] = 1.0;
            }
        }
    }

    std::vector<int64_t> counts = counts_from_pdf_flat(normalized.data(), len, fp_scale);
    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t sum = 0;
        for (int64_t row = 0; row < n_bins; ++row) {
            sum += counts[static_cast<size_t>(row * n_cols + col)];
        }
        if (sum <= 0) {
            for (int64_t row = 0; row < n_bins; ++row) {
                counts[static_cast<size_t>(row * n_cols + col)] = 1;
            }
        }
    }

    std::vector<int64_t> cdf(static_cast<size_t>(len), 0);
    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t num_sum = 0;
        for (int64_t row = 0; row < n_bins; ++row) {
            num_sum += counts[static_cast<size_t>(row * n_cols + col)];
        }
        TORCH_CHECK(num_sum > 0, "invalid zero-count column");

        std::vector<int64_t> base(static_cast<size_t>(n_bins), 0);
        int64_t base_sum = 0;
        for (int64_t row = 0; row < n_bins; ++row) {
            const int64_t num = counts[static_cast<size_t>(row * n_cols + col)];
            const int64_t value = (alloc * num) / num_sum;
            base[static_cast<size_t>(row)] = value;
            base_sum += value;
        }

        const int64_t remainder = alloc - base_sum;
        if (remainder > 0) {
            std::vector<std::pair<int64_t, int64_t>> order;
            order.reserve(static_cast<size_t>(n_bins));
            for (int64_t row = 0; row < n_bins; ++row) {
                const int64_t num = counts[static_cast<size_t>(row * n_cols + col)];
                const int64_t prio = (alloc * num) - (num_sum * base[static_cast<size_t>(row)]);
                const int64_t key = prio * (n_bins + 1) - row;
                order.emplace_back(key, row);
            }
            std::sort(order.begin(), order.end(), std::greater<>());
            for (int64_t idx = 0; idx < remainder; ++idx) {
                base[static_cast<size_t>(order[static_cast<size_t>(idx)].second)] += 1;
            }
        }

        int64_t running = 0;
        for (int64_t row = 0; row < n_bins; ++row) {
            running += base[static_cast<size_t>(row)] + min_range;
            cdf[static_cast<size_t>(row * n_cols + col)] = running;
        }
        TORCH_CHECK(running == total, "cdf sum mismatch");
    }

    return cdf;
}

class BitWriter {
public:
    void push_bit(uint8_t bit) {
        current_value_ += static_cast<uint64_t>(bit) << current_bits_;
        ++current_bits_;
        while (current_bits_ >= 8) {
            const auto lower = static_cast<uint8_t>(current_value_ & 0xff);
            current_bits_ -= 8;
            current_value_ >>= 8;
            bytes_.push_back(lower);
        }
    }

    std::string finish() {
        if (current_bits_ > 0) {
            bytes_.push_back(static_cast<uint8_t>(current_value_));
            current_value_ = 0;
            current_bits_ = 0;
        }
        return std::string(bytes_.begin(), bytes_.end());
    }

private:
    uint64_t current_value_ = 0;
    uint8_t current_bits_ = 0;
    std::vector<uint8_t> bytes_;
};

class BitReader {
public:
    explicit BitReader(std::vector<uint8_t> data)
        : data_(std::move(data)) {}

    bool pull_bit(uint8_t& bit) {
        while (current_bits_ < 1) {
            if (offset_ >= data_.size()) {
                return false;
            }
            const auto byte = data_[offset_++];
            current_value_ += static_cast<uint64_t>(byte) << current_bits_;
            current_bits_ += 8;
        }
        bit = static_cast<uint8_t>(current_value_ & 1);
        current_value_ >>= 1;
        --current_bits_;
        return true;
    }

private:
    std::vector<uint8_t> data_;
    size_t offset_ = 0;
    uint64_t current_value_ = 0;
    uint8_t current_bits_ = 0;
};

std::vector<uint8_t> bytes_to_vec(const py::bytes& data) {
    const std::string raw = data;
    return std::vector<uint8_t>(raw.begin(), raw.end());
}

class ArithmeticEncoder {
public:
    explicit ArithmeticEncoder(int64_t total_range_bits = 24)
        : total_range_bits_(total_range_bits) {
        TORCH_CHECK(total_range_bits_ <= 30, "total_range_bits must be <= 30");
    }

    void push_pdf_symbols_torch(
        const torch::Tensor& pdf_mat,
        const torch::Tensor& symbols,
        int64_t fp_scale,
        int64_t min_range
    ) {
        check_pdf_mat(pdf_mat);
        const auto n_bins = pdf_mat.size(0);
        const auto n_cols = pdf_mat.size(1);
        check_symbol_tensor(symbols, n_cols, "symbols");

        const auto* pdf = pdf_mat.data_ptr<double>();
        const auto* symbol_ptr = symbols.data_ptr<int64_t>();
        const auto cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            total_range_bits_,
            fp_scale,
            min_range
        );
        for (int64_t col = 0; col < n_cols; ++col) {
            TORCH_CHECK(symbol_ptr[col] >= 0, "symbols must be non-negative");
            push_symbol(static_cast<size_t>(symbol_ptr[col]), cdf, n_bins, n_cols, col);
        }
    }

    py::bytes finish() {
        while (max_bit_ >= 0) {
            const auto bit = static_cast<uint8_t>((low_ >> max_bit_) & 1);
            writer_.push_bit(bit);
            --max_bit_;
        }
        return py::bytes(writer_.finish());
    }

private:
    uint64_t delta() const {
        return high_ - low_ + 1;
    }

    void flush_common_prefix() {
        while (max_bit_ >= 0) {
            const auto b1 = low_ >> max_bit_;
            const auto b2 = high_ >> max_bit_;
            if (b1 == b2) {
                low_ -= b1 << max_bit_;
                high_ -= b1 << max_bit_;
                --max_bit_;
                writer_.push_bit(static_cast<uint8_t>(b1));
            } else {
                break;
            }
        }
    }

    void push_symbol(
        size_t symbol,
        const std::vector<int64_t>& cdf,
        int64_t n_bins,
        int64_t n_cols,
        int64_t col
    ) {
        while (delta() < (uint64_t{1} << total_range_bits_)) {
            low_ <<= 1;
            high_ = (high_ << 1) | 1;
            ++max_bit_;
        }
        TORCH_CHECK(static_cast<int64_t>(symbol) < n_bins, "symbol out of range");
        const auto total = uint64_t{1} << total_range_bits_;
        const auto rng = delta();
        const auto cum_high = static_cast<uint64_t>(cdf[symbol * static_cast<size_t>(n_cols) + static_cast<size_t>(col)]);
        const auto cum_low = symbol == 0
            ? 0
            : static_cast<uint64_t>(cdf[(symbol - 1) * static_cast<size_t>(n_cols) + static_cast<size_t>(col)]);
        const auto base = low_;
        low_ = base + (rng * cum_low) / total;
        high_ = base + (rng * cum_high) / total - 1;
        flush_common_prefix();
    }

    int64_t total_range_bits_;
    uint64_t low_ = 0;
    uint64_t high_ = 0;
    int64_t max_bit_ = -1;
    BitWriter writer_;
};

class ArithmeticDecoder {
public:
    explicit ArithmeticDecoder(py::bytes data, int64_t total_range_bits = 24)
        : total_range_bits_(total_range_bits),
          reader_(bytes_to_vec(data)) {
        TORCH_CHECK(total_range_bits_ <= 30, "total_range_bits must be <= 30");
    }

    void pull_symbols_into_torch(
        const torch::Tensor& pdf_mat,
        torch::Tensor out_symbols,
        int64_t fp_scale,
        int64_t min_range
    ) {
        check_pdf_mat(pdf_mat);
        const auto n_bins = pdf_mat.size(0);
        const auto n_cols = pdf_mat.size(1);
        check_symbol_tensor(out_symbols, n_cols, "out_symbols");

        const auto* pdf = pdf_mat.data_ptr<double>();
        auto* out_ptr = out_symbols.data_ptr<int64_t>();
        const auto cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            total_range_bits_,
            fp_scale,
            min_range
        );
        for (int64_t col = 0; col < n_cols; ++col) {
            out_ptr[col] = static_cast<int64_t>(pull_symbol(cdf, n_bins, n_cols, col));
        }
    }

private:
    uint64_t delta() const {
        return high_ - low_ + 1;
    }

    void flush_common_prefix() {
        while (max_bit_ >= 0) {
            const auto b1 = low_ >> max_bit_;
            const auto b2 = high_ >> max_bit_;
            if (b1 == b2) {
                low_ -= b1 << max_bit_;
                high_ -= b1 << max_bit_;
                current_ -= b1 << max_bit_;
                --max_bit_;
            } else {
                break;
            }
        }
    }

    size_t pull_symbol(
        const std::vector<int64_t>& cdf,
        int64_t n_bins,
        int64_t n_cols,
        int64_t col
    ) {
        while (delta() < (uint64_t{1} << total_range_bits_)) {
            uint8_t bit = 0;
            TORCH_CHECK(reader_.pull_bit(bit), "stream exhausted");
            low_ <<= 1;
            high_ = (high_ << 1) | 1;
            current_ = (current_ << 1) | static_cast<uint64_t>(bit);
            ++max_bit_;
        }

        const auto total = uint64_t{1} << total_range_bits_;
        const auto rng = delta();
        const auto target = (((current_ - low_ + 1) * total) - 1) / rng;

        int64_t lo = 0;
        int64_t hi = n_bins;
        while (lo < hi) {
            const auto mid = (lo + hi) / 2;
            const auto value = static_cast<uint64_t>(cdf[mid * n_cols + col]);
            if (target < value) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        TORCH_CHECK(lo < n_bins, "binary search failed");

        const auto symbol = static_cast<size_t>(lo);
        const auto cum_high = static_cast<uint64_t>(cdf[symbol * static_cast<size_t>(n_cols) + static_cast<size_t>(col)]);
        const auto cum_low = symbol == 0
            ? 0
            : static_cast<uint64_t>(cdf[(symbol - 1) * static_cast<size_t>(n_cols) + static_cast<size_t>(col)]);
        const auto base = low_;
        low_ = base + (rng * cum_low) / total;
        high_ = base + (rng * cum_high) / total - 1;
        flush_common_prefix();
        return symbol;
    }

    int64_t total_range_bits_;
    uint64_t low_ = 0;
    uint64_t high_ = 0;
    uint64_t current_ = 0;
    int64_t max_bit_ = -1;
    BitReader reader_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ArithmeticEncoder>(m, "ArithmeticEncoder")
        .def(py::init<int64_t>(), py::arg("total_range_bits") = 24)
        .def("push_pdf_symbols_torch", &ArithmeticEncoder::push_pdf_symbols_torch)
        .def("finish", &ArithmeticEncoder::finish);

    py::class_<ArithmeticDecoder>(m, "ArithmeticDecoder")
        .def(py::init<py::bytes, int64_t>(), py::arg("data"), py::arg("total_range_bits") = 24)
        .def("pull_symbols_into_torch", &ArithmeticDecoder::pull_symbols_into_torch);
}
