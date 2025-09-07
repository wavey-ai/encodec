# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Arithmetic coder."""

import io
import math
import random
import typing as tp
import torch

from ..binary import BitPacker, BitUnpacker

# encodec/quantization/ac.py — build_stable_quantized_cdf
def build_stable_quantized_cdf(pdf: torch.Tensor, total_range_bits: int,
                               fp_scale: int = 1 << 16, min_range: int = 2,
                               check: bool = True) -> torch.Tensor:
    pdf = pdf.detach().to(torch.float64).clamp_min(0)
    s = pdf.sum()
    if not torch.isfinite(s) or s <= 0:
        pdf = torch.ones_like(pdf)
        s = pdf.sum()

    # --- key change: avoid round-to-nearest; floor in fp64 then distribute remainder deterministically
    num = torch.floor(pdf * fp_scale).to(torch.int64)
    if num.sum() <= 0:
        num = torch.ones_like(num)

    total = 1 << total_range_bits
    n = int(num.numel())
    alloc = total - min_range * n
    num_sum = num.sum()

    base = (alloc * num) // num_sum
    remainder = int(alloc - int(base.sum().item()))
    if remainder > 0:
        idx = torch.arange(n, device=num.device, dtype=torch.int64)
        prio = (alloc * num) - (num_sum * base)
        key = prio * (n + 1) - idx        # deterministic tie-breaker
        _, order = torch.sort(key, descending=True)
        take = order[:remainder]
        base[take] += 1

    ranges = base + min_range
    cdf = torch.cumsum(ranges, dim=-1, dtype=torch.int64)

    if check:
        if int(cdf[-1].item()) != total:
            raise ValueError("cdf sum mismatch")
        if (ranges < min_range).any():
            raise ValueError("min_range violated")
    return cdf

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
        # If self.low and self.high start with the sames bits,
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
            quantized_cdf (torch.Tensor): use `build_stable_quantized_cdf`
                to build this from your pdf estimate.
        """
        while self.delta < 2 ** self.total_range_bits:
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.max_bit += 1
        total = 1 << self.total_range_bits
        rng = self.delta
        cum_low = 0 if symbol == 0 else int(quantized_cdf[symbol - 1].item())
        cum_high = int(quantized_cdf[symbol].item())
        base = self.low
        new_low = base + (rng * cum_low) // total
        new_high = base + (rng * cum_high) // total - 1
        self.low = new_low
        self.high = new_high
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
    """ArithmeticDecoder, see `ArithmeticCoder` for a detailed explanation.

    Note that this must be called with **exactly** the same parameters and sequence
    of quantized cdf as the arithmetic encoder or the wrong values will be decoded.

    If the AC encoder current range is [L, H], with `L` and `H` having the some common
    prefix (i.e. the same most significant bits), then this prefix will be flushed to the stream.
    For instances, having read 3 bits `b1 b2 b3`, we know that `[L, H]` is contained inside
    `[b1 b2 b3 0 ... 0 b1 b3 b3 1 ... 1]`. Now this specific sub-range can only be obtained
    for a specific sequence of symbols and a binary-search allows us to decode those symbols.
    At some point, the prefix `b1 b2 b3` will no longer be sufficient to decode new symbols,
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
        This returns `None` when the stream has been exhausted.

        Args:
            quant[48;48;201;1632;2814tized_cdf (torch.Tensor): use `build_stable_quantized_cdf`
                to build this from your pdf estimate. This must be **exatly**
                the same cdf as the one used at encoding time.
        """
        while self.delta < 2 ** self.total_range_bits:
            bit = self.unpacker.pull()
            if bit is None:
                return None
            self.low = self.low << 1
            self.high = (self.high << 1) | 1
            self.current = (self.current << 1) | bit
            self.max_bit += 1

        total = 1 << self.total_range_bits
        rng = self.delta
        target = ((self.current - self.low + 1) * total - 1) // rng
        t = torch.tensor(target, dtype=quantized_cdf.dtype, device=quantized_cdf.device)
        s = torch.searchsorted(quantized_cdf, t, right=True).item()
        cum_low = 0 if s == 0 else int(quantized_cdf[s - 1].item())
        cum_high = int(quantized_cdf[s].item())
        base = self.low
        self.low = base + (rng * cum_low) // total
        self.high = base + (rng * cum_high) // total - 1
        self._dbg.append((self.low, self.high, self.current))
        self._flush_common_prefix()
        self._dbg2.append((self.low, self.high, self.current))

        return s


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
