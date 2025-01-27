# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Arithmetic coder."""

import typing as tp
import io
import random
import torch
import numpy as np

print(torch.__config__.show())

from ..binary import BitPacker, BitUnpacker

# Define the fixed-point scaling factor
FIXED_SCALE = 1 << 32

def pdf_to_fixed_point(pdf: torch.Tensor) -> torch.Tensor:
    """
    Converts a floating-point PDF to fixed-point integer representation 
    while eliminating floating-point precision drift.
    """
    assert torch.all(pdf >= 0), "PDF contains negative values!"
    
    # Ensure pdf is on CPU and double precision before conversion
    pdf = pdf.to(torch.float64).cpu()
    
    # Perform scaling using integer arithmetic only
    scaled_pdf = (pdf * FIXED_SCALE + 0.5).floor().to(torch.int64)

    return scaled_pdf


def deterministic_round(x: torch.Tensor) -> torch.Tensor:
    """
    Implements a deterministic rounding method: rounds half up.
    """
    return torch.floor(x + 0.5)

def pdf_to_integer_counts_fixed(pdf_fixed: torch.Tensor, total_range: int) -> torch.Tensor:
    """
    Converts a fixed-point PDF into integer counts that sum to total_range using integer-only operations.
    
    Args:
        pdf_fixed (torch.Tensor): Fixed-point integer representation of the PDF.
        total_range (int): Desired sum of the integer counts.
    
    Returns:
        torch.Tensor: Integer counts summing to total_range.
    """
    assert torch.all(pdf_fixed >= 0), "PDF contains negative values!"
    assert torch.isclose(pdf_fixed.sum(), torch.tensor(FIXED_SCALE, dtype=pdf_fixed.dtype, device=pdf_fixed.device), atol=1), "PDF does not sum to fixed scale!"
 
    # Step 1: Scale the PDF using total_range and perform integer division
    scaled_pdf = (pdf_fixed * total_range + (FIXED_SCALE // 2)) // FIXED_SCALE
    
    # Step 2: Calculate the sum of the scaled PDF
    current_sum = scaled_pdf.sum().item()
    deficit = total_range - current_sum
    
    # Step 3: Redistribute the deficit deterministically
    if deficit > 0:
        fractional_remainders = (pdf_fixed * total_range) % FIXED_SCALE
        # Tie-breaking using indices to ensure deterministic sorting
        sorted_indices = torch.argsort(
            fractional_remainders * 1_000_000 + torch.arange(len(fractional_remainders), dtype=torch.int64, device=pdf_fixed.device),
            descending=True
        )
        selected_indices = sorted_indices[:deficit]
        scaled_pdf[selected_indices] += 1
    
    elif deficit < 0:
        # Need to remove excess counts
        excess = -deficit
        sorted_indices = torch.argsort(scaled_pdf, descending=True)
        scaled_pdf[sorted_indices[:excess]] -= 1
    
    return scaled_pdf

def build_stable_quantized_cdf(pdf: torch.Tensor, total_range_bits: int,
                               min_range: int = 2, check: bool = True) -> torch.Tensor:
    """Integer-only version of build_stable_quantized_cdf that avoids floating point operations."""

    total_range = 1 << total_range_bits
    cardinality = len(pdf)

    pdf_fixed = pdf_to_fixed_point(pdf)

    counts = pdf_to_integer_counts_fixed(pdf_fixed, total_range)

    deficit = min_range * cardinality - counts.sum().item()

    if deficit > 0:
        available = counts - min_range
        available_total = available[available > 0].sum().item()

        if available_total > 0:
            reduction = (available * deficit).div(available_total, rounding_mode='floor')
            counts[available > 0] -= reduction
            deficit -= reduction.sum().item()

        if deficit > 0:
            per_symbol = deficit // cardinality
            remainder = deficit % cardinality
            counts += per_symbol
            counts[:remainder] += 1

    counts = torch.maximum(counts, torch.tensor(min_range))

    # Scale down if we exceed total range
    if counts.sum().item() > total_range:
        scale = total_range / counts.sum().item()
        counts = (counts * scale).round().long()
        counts = torch.maximum(counts, torch.tensor(min_range))
        counts[counts.argmax()] -= (counts.sum().item() - total_range)

    # Build CDF through cumulative sum
    quantized_cdf = torch.cumsum(counts, dim=-1)

    if check:
        assert quantized_cdf[-1].item() <= 2 ** total_range_bits, f"CDF exceeds range: {quantized_cdf[-1]}"
        ranges = torch.diff(quantized_cdf, prepend=torch.tensor([0]))
        assert (ranges >= min_range).all(), f"Some ranges below minimum: {ranges.min().item()}"

    #sys.exit(1)  # Exit immediately after logging first case


    log_file = "quantized_cdf_log.txt"  # Log file path
    with open(log_file, "a") as f:  # Corrected syntax
        f.write(str(quantized_cdf) + "\n")

    return quantized_cdf


def compute_effective_range(range_low: int, range_high: int, delta: int, total_range_bits: int) -> tp.Tuple[int, int]:
    total_range = 1 << total_range_bits

    # Scale delta using fixed-point scaling
    scaled_delta = delta * FIXED_SCALE

    # Compute effective_low and effective_high with fixed-point precision
    effective_low_fixed = (range_low * scaled_delta + (total_range // 2)) // total_range
    effective_high_fixed = (range_high * scaled_delta) // total_range

    # Convert back from fixed-point to integer
    effective_low = effective_low_fixed // FIXED_SCALE
    effective_high = effective_high_fixed // FIXED_SCALE

    # Ensure that effective_high is at least effective_low to maintain a valid range
    effective_high = max(effective_high, effective_low)

    return effective_low, effective_high

class ArithmeticCoder:
    """ArithmeticCoder,
    Let us take a distribution `p` over `N` symbols, and assume we have a stream
    of random variables `s_t` sampled from `p`. Let us assume that we have a budget
    of `B` bits that we can afford to write on device. There are `2**B` possible numbers,
    corresponding to the range `[0, 2 ** B - 1]`. We can map each of those number to a single
    sequence `(s_t)` by doing the following:

    1) Initialize the current range to` [0 ** 2 B - 1]`.
    2) For each time step t, split the current range into contiguous chunks,
        one for each possible outcome, with size roughly proportional to `p`.
        For instance, if `p = [0.75, 0.25]`, and the range is `[0, 3]`, the chunks
        would be `{[0, 2], [3, 3]}`.
    3) Select the chunk corresponding to `s_t`, and replace the current range with this.
    4) When done encoding all the values, just select any value remaining in the range.

    You will notice that this procedure can fail: for instance if at any point in time
    the range is smaller than `N`, then we can no longer assign a non-empty chunk to each
    possible outcome. Intuitively, the more likely a value is, the less the range width
    will reduce, and the longer we can go on encoding values. This makes sense: for any efficient
    coding scheme, likely outcomes would take less bits, and more of them can be coded
    with a fixed budget.

    In practice, we do not know `B` ahead of time, but we have a way to inject new bits
    when the current range decreases below a given limit (given by `total_range_bits`), without
    having to redo all the computations. If we encode mostly likely values, we will seldom
    need to inject new bits, but a single rare value can deplete our stock of entropy!

    In this explanation, we assumed that the distribution `p` was constant. In fact, the present
    code works for any sequence `(p_t)` possibly different for each timestep.
    We also assume that `s_t ~ p_t`, but that doesn't need to be true, although the smaller
    the KL between the true distribution and `p_t`, the most efficient the coding will be.

    Args:
        fo (IO[bytes]): file-like object to which the bytes will be written to.
        total_range_bits (int): the range `M` described above is `2 ** total_range_bits.
            Any time the current range width fall under this limit, new bits will
            be injected to rescale the initial range.
    """

    def __init__(self, fo: tp.IO[bytes], total_range_bits: int = 24):
        assert total_range_bits <= 30
        self.total_range_bits = total_range_bits
        self.packer = BitPacker(bits=1, fo=fo)  # we push single bits at a time.
        self.low: int = 0
        self.high: int = 0
        self.max_bit: int = -1
        self._dbg: tp.List[tp.Any] = []
        self._dbg2: tp.List[tp.Any] = []

    @property
    def delta(self) -> int:
        """Return the current range width."""
        return self.high - self.low + 1

    def _flush_common_prefix(self):
        # If self.low and self.high start with the same bits,
        # those won't change anymore as we always just increase the range
        # by powers of 2, and we can flush them out to the bit stream.
        assert self.high >= self.low, (self.low, self.high)
        assert self.high < 2 ** (self.max_bit + 1)
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            b2 = self.high >> self.max_bit
            if b1 == b2:
                self.low -= (b1 << self.max_bit)
                self.high -= (b1 << self.max_bit)
                assert self.high >= self.low, (self.high, self.low, self.max_bit)
                assert self.low >= 0
                self.max_bit -= 1
                self.packer.push(b1)
            else:
                break

    def push(self, symbol: int, quantized_cdf: torch.Tensor):
        """Push the given symbol on the stream, flushing out bits
        if possible.

        Args:
            symbol (int): symbol to encode with the AC.
            quantized_cdf (torch.Tensor): use build_stable_quantized_cdf
                to build this from your pdf estimate.
        """
        while self.delta < (1 << self.total_range_bits):
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.max_bit += 1

        range_low = 0 if symbol == 0 else quantized_cdf[symbol - 1].item()
        range_high = quantized_cdf[symbol].item() - 1
        effective_low, effective_high = compute_effective_range(range_low, range_high, self.delta, self.total_range_bits)
        assert self.low <= self.high
        self.high = self.low + effective_high
        self.low = self.low + effective_low
        assert self.low <= self.high, (effective_low, effective_high, range_low, range_high)
        self._dbg.append((self.low, self.high))
        self._dbg2.append((self.low, self.high))
        outs = self._flush_common_prefix()
        assert self.low <= self.high
        assert self.max_bit >= -1
        assert self.max_bit <= 61, self.max_bit
        return outs

    def flush(self):
        """Flush the remaining information to the stream.
        """
        while self.max_bit >= 0:
            b1 = (self.low >> self.max_bit) & 1
            self.packer.push(b1)
            self.max_bit -= 1
        self.packer.flush()


class ArithmeticDecoder:
    """ArithmeticDecoder, see ArithmeticCoder for a detailed explanation.

    Note that this must be called with **exactly** the same parameters and sequence
    of quantized cdf as the arithmetic encoder or the wrong values will be decoded.

    If the AC encoder current range is [L, H], with L and H having the some common
    prefix (i.e. the same most significant bits), then this prefix will be flushed to the stream.
    For instances, having read 3 bits b1 b2 b3, we know that [L, H] is contained inside
    [b1 b2 b3 0 ... 0 b1 b3 b3 1 ... 1]. Now this specific sub-range can only be obtained
    for a specific sequence of symbols and a binary-search allows us to decode those symbols.
    At some point, the prefix b1 b2 b3 will no longer be sufficient to decode new symbols,
    and we will need to read new bits from the stream and repeat the process.

    """
    def __init__(self, fo: tp.IO[bytes], total_range_bits: int = 24):
        self.total_range_bits = total_range_bits
        self.low: int = 0
        self.high: int = 0
        self.current: int = 0
        self.max_bit: int = -1
        self.unpacker = BitUnpacker(bits=1, fo=fo)  # we pull single bits at a time.
        # Following is for debugging
        self._dbg: tp.List[tp.Any] = []
        self._dbg2: tp.List[tp.Any] = []
        self._last: tp.Any = None

    @property
    def delta(self) -> int:
        return self.high - self.low + 1

    def _flush_common_prefix(self):
        # Given the current range [L, H], if both have a common prefix,
        # we know we can remove it from our representation to avoid handling large numbers.
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            b2 = self.high >> self.max_bit
            if b1 == b2:
                self.low -= (b1 << self.max_bit)
                self.high -= (b1 << self.max_bit)
                self.current -= (b1 << self.max_bit)
                assert self.high >= self.low
                assert self.low >= 0
                self.max_bit -= 1
            else:
                break

    def pull(self, quantized_cdf: torch.Tensor) -> tp.Optional[int]:
        """Pull a symbol, reading as many bits from the stream as required.
        This returns None when the stream has been exhausted.

        Args:
            quantized_cdf (torch.Tensor): use build_stable_quantized_cdf
                to build this from your pdf estimate. This must be **exactly**
                the same cdf as the one used at encoding time.
        """
        while self.delta < (1 << self.total_range_bits):
            bit = self.unpacker.pull()
            if bit is None:
                return None
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.current = (self.current << 1) | bit
            self.max_bit += 1

        log_file = "binary_search_log.txt"  # Log file path
        with open(log_file, "a") as f:

            def bin_search(low_idx: int, high_idx: int):
                if high_idx < low_idx:
                    raise RuntimeError("Binary search failed")
                mid = (low_idx + high_idx) // 2
                range_low = quantized_cdf[mid - 1].item() if mid > 0 else 0
                range_high = quantized_cdf[mid].item() - 1
                effective_low, effective_high = compute_effective_range(range_low, range_high, self.delta, self.total_range_bits)
                low = effective_low + self.low
                high = effective_high + self.low

                # Log each iteration
                f.write(f"low_idx={low_idx}, high_idx={high_idx}, mid={mid}, low={low}, high={high}, current={self.current}\n")

                if self.current >= low:
                    if self.current <= high:
                        return (mid, low, high, self.current)
                    else:
                        return bin_search(mid + 1, high_idx)
                else:
                    return bin_search(low_idx, mid - 1)

            self._last = (self.low, self.high, self.current, self.max_bit)
            sym, self.low, self.high, self.current = bin_search(0, len(quantized_cdf) - 1)
            self._dbg.append((self.low, self.high, self.current))
            self._flush_common_prefix()
            self._dbg2.append((self.low, self.high, self.current))

            return sym

    def pullx(self, quantized_cdf: torch.Tensor) -> tp.Optional[int]:
        """Pull a symbol, reading as many bits from the stream as required.
        This returns None when the stream has been exhausted.

        Args:
            quantized_cdf (torch.Tensor): use build_stable_quantized_cdf
                to build this from your pdf estimate. This must be **exactly**
                the same cdf as the one used at encoding time.
        """
        while self.delta < (1 << self.total_range_bits):
            bit = self.unpacker.pull()
            if bit is None:
                return None
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.current = (self.current << 1) | bit
            self.max_bit += 1

        def bin_search(low_idx: int, high_idx: int):
            # Binary search is not just for coding interviews :)
            if high_idx < low_idx:
                raise RuntimeError("Binary search failed")
            mid = (low_idx + high_idx) // 2
            range_low = quantized_cdf[mid - 1].item() if mid > 0 else 0
            range_high = quantized_cdf[mid].item() - 1
            effective_low, effective_high = compute_effective_range(range_low, range_high, self.delta, self.total_range_bits)
            low = effective_low + self.low
            high = effective_high + self.low
            if self.current >= low:
                if self.current <= high:
                    return (mid, low, high, self.current)
                else:
                    return bin_search(mid + 1, high_idx)
            else:
                return bin_search(low_idx, mid - 1)

        self._last = (self.low, self.high, self.current, self.max_bit)
        sym, self.low, self.high, self.current = bin_search(0, len(quantized_cdf) - 1)
        self._dbg.append((self.low, self.high, self.current))
        self._flush_common_prefix()
        self._dbg2.append((self.low, self.high, self.current))

        return sym


def test():
    torch.manual_seed(1234)
    random.seed(1234)
    for _ in range(4):
        pdfs = []
        cardinality = random.randrange(4000)
        steps = random.randrange(100, 500)
        fo = io.BytesIO()
        encoder = ArithmeticCoder(fo)
        symbols = []
        for step in range(steps):
            pdf = torch.softmax(torch.randn(cardinality), dim=0)
            pdfs.append(pdf)
            q_cdf = build_stable_quantized_cdf(pdf, encoder.total_range_bits)
            symbol = torch.multinomial(pdf, 1).item()
            symbols.append(symbol)
            encoder.push(symbol, q_cdf)
        encoder.flush()

        fo.seek(0)
        decoder = ArithmeticDecoder(fo)
        for idx, (pdf, symbol) in enumerate(zip(pdfs, symbols)):
            q_cdf = build_stable_quantized_cdf(pdf, encoder.total_range_bits)
            decoded_symbol = decoder.pull(q_cdf)
            assert decoded_symbol == symbol, idx
        assert decoder.pull(torch.zeros(1)) is None


if __name__ == "__main__":
    test()

