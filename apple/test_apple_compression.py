#!/usr/bin/env python3

# pyright: strict

import abc
import base64
import collections.abc
import lzma
import random
import sys
import typing
import zlib

import pytest

from apple_compression import *


class _InteroperableImplementation(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def compress(data: bytes) -> bytes:
        ...

    @staticmethod
    @abc.abstractmethod
    def decompress(data: bytes) -> bytes:
        ...


# For interoperability testing, use raw zlib (no header or trailing checksum),
# as that’s what libcompression implements.
class _InteroperableZlibRaw(_InteroperableImplementation):

    @staticmethod
    @typing.override
    def compress(data: bytes) -> bytes:
        return zlib.compress(data, wbits=-zlib.MAX_WBITS)

    @staticmethod
    @typing.override
    def decompress(data: bytes) -> bytes:
        return zlib.decompress(data, wbits=-zlib.MAX_WBITS)


class _InteroperableLzma(_InteroperableImplementation):

    @staticmethod
    @typing.override
    def compress(data: bytes) -> bytes:
        return lzma.compress(data)

    @staticmethod
    @typing.override
    def decompress(data: bytes) -> bytes:
        return lzma.decompress(data)


_interoperable_algorithms = {
    Algorithm.ZLIB: _InteroperableZlibRaw,
    Algorithm.LZMA: _InteroperableLzma,
}

_no_stream_algorithms = (
    Algorithm.LZ4_RAW,
    Algorithm.LZBITMAP,
)

# These decompress to empty output (b'').
_compressed_empty = {
    Algorithm.LZ4: 'YnY0JA==',
    Algorithm.ZLIB: 'AwA=',
    Algorithm.LZMA: '/Td6WFoAAAD/EtlBAAAAABzfRCEGcp56AQAAAAAAWVo=',
    Algorithm.LZ4_RAW: '',
    Algorithm.BROTLI: 'Bg==',
    Algorithm.LZFSE: 'YnZ4LQAAAABidngk',
    Algorithm.LZBITMAP: 'WkJNCQYAAAAAAA==',
}

# These decompress to a NUL byte (b'\x00').
_compressed_nul = {
    Algorithm.LZ4: 'YnY0LQEAAAAAYnY0JA==',
    Algorithm.ZLIB: 'YwAA',
    Algorithm.LZMA: '/Td6WFoAAAD/EtlBAsAFASEBFgAn6GODAQAAAAAAAAAAAREB'
                    'raZYBAZynnoBAAAAAABZWg==',
    Algorithm.LZ4_RAW: 'EAA=',
    Algorithm.BROTLI: 'CwCAAAM=',
    Algorithm.LZFSE: 'YnZ4LQEAAAAAYnZ4JA==',
    Algorithm.LZBITMAP: 'WkJNCQcAAAEAAAAGAAAAAAA=',
}

# These decompress to (b'test ' * 16) (80 bytes).
_compressed_compressible = {
    Algorithm.LZ4: 'YnY0LVAAAAB0ZXN0IHRlc3QgdGVzdCB0ZXN0IHRlc3QgdGVz'
                   'dCB0ZXN0IHRlc3QgdGVzdCB0ZXN0IHRlc3QgdGVzdCB0ZXN0'
                   'IHRlc3QgdGVzdCB0ZXN0IGJ2NCQ=',
    Algorithm.ZLIB: 'K0ktLlEooR4BAA==',
    Algorithm.LZMA: '/Td6WFoAAAD/EtlBAsATUCEBFgDa759T4ABPAAtdADoZSs4c'
                    '/Aet+oAAAAAAAR9QQeq3hgZynnoBAAAAAABZWg==',
    Algorithm.LZ4_RAW: '8EF0ZXN0IHRlc3QgdGVzdCB0ZXN0IHRlc3QgdGVzdCB0ZXN0'
                       'IHRlc3QgdGVzdCB0ZXN0IHRlc3QgdGVzdCB0ZXN0IHRlc3Qg'
                       'dGVzdCB0ZXN0IA==',
    Algorithm.BROTLI: 'G08AAKBByuboDMDBWBCJSWKAO21WlG1Q',
    Algorithm.LZFSE: 'YnZ4blAAAAAVAAAA5XRlc3QgOAXwL+J0IAYAAAAAAAAAYnZ4JA==',
    Algorithm.LZBITMAP: 'WkJNCVYAAFAAAHRlc3QgdGVzdCB0ZXN0IHRlc3QgdGVzdCB0'
                        'ZXN0IHRlc3QgdGVzdCB0ZXN0IHRlc3QgdGVzdCB0ZXN0IHRl'
                        'c3QgdGVzdCB0ZXN0IHRlc3QgBgAAAAAA',
}

# These decompress to 64 random bytes.
_incompressible_compressed = base64.b64decode(
    'oXSRaY4VSGbPiFrHsv8+d78fqznxdqEfr+E4lcPcPJ/Jbs2d'
    'Rhj/yqBdydXFmDEcHjfJWbB+pm3VJkXXlOtv2Q==')
_compressed_incompressible = {
    Algorithm.LZ4: 'YnY0LUAAAAChdJFpjhVIZs+IWsey/z53vx+rOfF2oR+v4TiV'
                   'w9w8n8luzZ1GGP/KoF3J1cWYMRweN8lZsH6mbdUmRdeU62/Z'
                   'YnY0JA==',
    Algorithm.ZLIB: 'AUAAv/+hdJFpjhVIZs+IWsey/z53vx+rOfF2oR+v4TiVw9w8'
                    'n8luzZ1GGP/KoF3J1cWYMRweN8lZsH6mbdUmRdeU62/Z',
    Algorithm.LZMA: '/Td6WFoAAAD/EtlBAsBEQCEBFgAXQPQiAQA/oXSRaY4VSGbP'
                    'iFrHsv8+d78fqznxdqEfr+E4lcPcPJ/Jbs2dRhj/yqBdydXF'
                    'mDEcHjfJWbB+pm3VJkXXlOtv2QAAAVBA76nh7AZynnoBAAAA'
                    'AABZWg==',
    Algorithm.LZ4_RAW: '8DGhdJFpjhVIZs+IWsey/z53vx+rOfF2oR+v4TiVw9w8n8lu'
                       'zZ1GGP/KoF3J1cWYMRweN8lZsH6mbdUmRdeU62/Z',
    Algorithm.BROTLI: 'ix+AoXSRaY4VSGbPiFrHsv8+d78fqznxdqEfr+E4lcPcPJ/J'
                      'bs2dRhj/yqBdydXFmDEcHjfJWbB+pm3VJkXXlOtv2QM=',
    Algorithm.LZFSE: 'YnZ4LUAAAAChdJFpjhVIZs+IWsey/z53vx+rOfF2oR+v4TiV'
                     'w9w8n8luzZ1GGP/KoF3J1cWYMRweN8lZsH6mbdUmRdeU62/Z'
                     'YnZ4JA==',
    Algorithm.LZBITMAP: 'WkJNCUYAAEAAAKF0kWmOFUhmz4hax7L/Pne/H6s58XahH6/h'
                        'OJXD3DyfyW7NnUYY/8qgXcnVxZgxHB43yVmwfqZt1SZF15Tr'
                        'b9kGAAAAAAA=',
}

_InputTypes = typing.Type[bytes | bytearray | memoryview]
_input_types = typing.get_args(typing.get_args(_InputTypes)[0])


# compress can’t distinguish between an error and a compression operation that
# doesn’t produce output.
@pytest.mark.parametrize('algorithm',
                         (algorithm if _compressed_empty[algorithm] != '' else
                          pytest.param(algorithm, marks=pytest.mark.xfail)
                          for algorithm in Algorithm))
def test_buffer_compress_empty(algorithm: Algorithm) -> None:
    expect = base64.b64decode(_compressed_empty[algorithm])
    compressed = compress(b'', algorithm)
    assert compressed == expect


@pytest.mark.parametrize('algorithm', Algorithm)
def test_buffer_decompress_nul(algorithm: Algorithm) -> None:
    compressed = base64.b64decode(_compressed_nul[algorithm])
    decompressed = decompress(compressed, algorithm)
    assert decompressed == b'\x00'


# For some reason, decompression produces an error using the buffer interface
# with this compressible data using Brotli.
@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm != Algorithm.BROTLI else
                          pytest.param(algorithm, marks=pytest.mark.xfail)
                          for algorithm in Algorithm))
def test_buffer_decompress_compressible(algorithm: Algorithm) -> None:
    compressed = base64.b64decode(_compressed_compressible[algorithm])
    decompressed = decompress(compressed, algorithm)
    assert decompressed == b'test ' * 16


@pytest.mark.parametrize('algorithm', Algorithm)
def test_buffer_decompress_incompressible(algorithm: Algorithm) -> None:
    compressed = base64.b64decode(_compressed_incompressible[algorithm])
    decompressed = decompress(compressed, algorithm)
    assert decompressed == _incompressible_compressed


@pytest.mark.parametrize('algorithm', Algorithm)
def test_buffer_decompress_sized(algorithm: Algorithm) -> None:
    compressed = base64.b64decode(_compressed_incompressible[algorithm])
    decompressed = decompress(compressed,
                              algorithm,
                              max_size=len(_incompressible_compressed))
    assert decompressed == _incompressible_compressed


@pytest.mark.parametrize('algorithm', Algorithm)
def test_buffer_decompress_sized_overflow(algorithm: Algorithm) -> None:
    compressed = base64.b64decode(_compressed_incompressible[algorithm])
    with pytest.raises(CompressionError):
        decompress(compressed,
                   algorithm,
                   max_size=len(_incompressible_compressed) - 1)


def _test_buffer_roundtrip(algorithm: Algorithm, data: bytes,
                           input_type: _InputTypes) -> None:
    data_t = input_type(data) if input_type is not bytes else data
    compressed = compress(data_t, algorithm)
    compressed_t = (input_type(compressed)
                    if input_type is not bytes else compressed)
    decompressed = decompress(compressed_t, algorithm)
    assert decompressed == data


@pytest.mark.parametrize('algorithm', Algorithm)
@pytest.mark.parametrize('input_type', _input_types)
def test_buffer_roundtrip_nul(algorithm: Algorithm,
                              input_type: _InputTypes) -> None:
    _test_buffer_roundtrip(algorithm, b'\x00', input_type)


# For some reason, decompression produces an error using the buffer interface
# with this compressible data using LZMA or Brotli.
@pytest.mark.parametrize(
    'algorithm',
    (algorithm if algorithm not in (Algorithm.LZMA, Algorithm.BROTLI) else
     pytest.param(algorithm, marks=pytest.mark.xfail)
     for algorithm in Algorithm))
@pytest.mark.parametrize('input_type', _input_types)
def test_buffer_roundtrip_compressible(algorithm: Algorithm,
                                       input_type: _InputTypes) -> None:
    _test_buffer_roundtrip(algorithm,
                           b''.join(c.to_bytes() for c in range(256)) * 4096,
                           input_type)


# This will almost certainly work with LZMA and Brotli, but because the data is
# random and they’ve been observed to fail decompression via the buffer
# interface on compressible data, there’s the possibility that this might fail
# if the random data turns out to be compressible. Skip those algorithms—don’t
# xfail them, because an xpass result would be undesirable.
@pytest.mark.parametrize(
    'algorithm',
    (algorithm if algorithm not in (Algorithm.LZMA, Algorithm.BROTLI) else
     pytest.param(algorithm, marks=pytest.mark.skip)
     for algorithm in Algorithm))
@pytest.mark.parametrize('input_type', _input_types)
def test_buffer_roundtrip_random(algorithm: Algorithm,
                                 input_type: _InputTypes) -> None:
    _test_buffer_roundtrip(algorithm, random.randbytes(1024 * 1024), input_type)


# Skip LZMA and Brotli for the same reason as in test_buffer_roundtrip_random.
@pytest.mark.parametrize(
    'algorithm',
    (algorithm if algorithm not in (Algorithm.LZMA, Algorithm.BROTLI) else
     pytest.param(algorithm, marks=pytest.mark.skip)
     for algorithm in Algorithm))
@pytest.mark.parametrize('input_type', _input_types)
def test_buffer_roundtrip_scratch(algorithm: Algorithm,
                                  input_type: _InputTypes) -> None:
    compress_scratch_size = compress_scratch_buffer_size(algorithm)
    assert compress_scratch_size >= 0
    decompress_scratch_size = decompress_scratch_buffer_size(algorithm)
    assert decompress_scratch_size >= 0

    data = random.randbytes(1024 * 1024)

    data_t = input_type(data) if input_type is not bytes else data
    compress_scratch_buffer = bytearray(compress_scratch_size)
    compress_scratch_buffer_t = (memoryview(compress_scratch_buffer)
                                 if input_type is memoryview else
                                 compress_scratch_buffer)
    compressed = compress(data_t,
                          algorithm,
                          scratch_buffer=compress_scratch_buffer_t)
    compressed_without_scratch = compress(data, algorithm)
    assert compressed == compressed_without_scratch

    compressed_t = (input_type(compressed)
                    if input_type is not bytes else compressed)
    decompress_scratch_buffer = bytearray(decompress_scratch_size)
    decompress_scratch_buffer_t = (memoryview(decompress_scratch_buffer)
                                   if input_type is memoryview else
                                   decompress_scratch_buffer)
    decompressed = decompress(compressed_t,
                              algorithm,
                              scratch_buffer=decompress_scratch_buffer_t)
    assert decompressed == data


# This test fails for some algorithms: compressing with zlib wants a max_size 1
# byte larger than will be populated; LZBITMAP needs somewhat more.
@pytest.mark.parametrize('algorithm', (algorithm if algorithm not in (
    Algorithm.ZLIB,
    Algorithm.LZBITMAP,
) else pytest.param(algorithm, marks=pytest.mark.xfail)
                                       for algorithm in Algorithm))
def test_buffer_compress_sized(algorithm: Algorithm) -> None:
    data = random.randbytes(1024 * 1024)
    compressed = compress(data, algorithm)
    compressed_sized = compress(data, algorithm, max_size=len(compressed))

    # LZFSE is weird: the implementation seems to compress more tightly when
    # given a smaller buffer. This means that `compressed` and `compressed_size`
    # may not be equal, but at least it’s possible to test that
    # `compressed_sized` decompresses properly.
    if algorithm != Algorithm.LZFSE:
        assert compressed == compressed_sized

    decompressed = decompress(compressed_sized, algorithm)
    assert decompressed == data


@pytest.mark.parametrize('algorithm', Algorithm)
def test_buffer_compress_sized_overflow(algorithm: Algorithm) -> None:
    data = random.randbytes(1024 * 1024)
    compressed = compress(data, algorithm)

    # LZFSE is weird: the implementation seems to compress more tightly when
    # given a smaller buffer. That means that undersizing the buffer by 1 byte
    # isn’t enough to trigger a CompressionError. Instead, undersize it by 2.5%,
    # which seems more than sufficient to trigger the error.
    max_size = (round(.975 * len(compressed))
                if algorithm == Algorithm.LZFSE else len(compressed) - 1)

    with pytest.raises(CompressionError):
        compress(data, algorithm, max_size=max_size)


@pytest.mark.parametrize('algorithm', _interoperable_algorithms.keys())
def test_buffer_decompress_interoperable(algorithm: Algorithm) -> None:
    data = random.randbytes(1024 * 1024)
    interoperable_implementation = _interoperable_algorithms[algorithm]
    compressed = interoperable_implementation.compress(data)
    decompressed = decompress(compressed, algorithm)
    assert decompressed == data


@pytest.mark.parametrize('algorithm', _interoperable_algorithms.keys())
def test_buffer_compress_interoperable(algorithm: Algorithm) -> None:
    data = random.randbytes(1024 * 1024)
    interoperable_implementation = _interoperable_algorithms[algorithm]
    compressed = compress(data, algorithm)
    decompressed = interoperable_implementation.decompress(compressed)
    assert decompressed == data


def _test_stream_decompress(
    algorithm: Algorithm,
    compressed_d: collections.abc.Mapping[Algorithm, str],
    expect: bytes,
) -> None:
    compressed = base64.b64decode(compressed_d[algorithm])
    with Decompressor(algorithm) as decompressor:
        assert not decompressor.eof
        assert decompressor.needs_input
        decompressed = decompressor.decompress(compressed)
        decompressed += decompressor.flush()
        assert decompressor.flush() == b''
        assert decompressed == expect
        assert decompressor.eof
        assert not decompressor.needs_input


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
def test_stream_decompress_empty(algorithm: Algorithm) -> None:
    _test_stream_decompress(algorithm, _compressed_empty, b'')


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
def test_stream_decompress_nul(algorithm: Algorithm) -> None:
    _test_stream_decompress(algorithm, _compressed_nul, b'\x00')


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
def test_stream_decompress_compressible(algorithm: Algorithm) -> None:
    _test_stream_decompress(algorithm, _compressed_compressible, b'test ' * 16)


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
def test_stream_decompress_incompressible(algorithm: Algorithm) -> None:
    _test_stream_decompress(algorithm, _compressed_incompressible,
                            _incompressible_compressed)


def _test_stream_roundtrip(algorithm: Algorithm, data: bytes,
                           input_type: _InputTypes) -> None:
    data_t = input_type(data) if input_type is not bytes else data

    with Compressor(algorithm) as compressor:
        compressed = compressor.compress(data_t)
        compressed += compressor.flush()

    compressed_t = (input_type(compressed)
                    if input_type is not bytes else compressed)

    with Decompressor(algorithm) as decompressor:
        assert not decompressor.eof
        assert decompressor.needs_input
        decompressed = decompressor.decompress(compressed_t)
        decompressed += decompressor.flush()
        assert decompressor.flush() == b''
        assert decompressed == data
        assert decompressor.eof
        assert not decompressor.needs_input


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
@pytest.mark.parametrize('input_type', _input_types)
def test_stream_roundtrip_empty(algorithm: Algorithm,
                                input_type: _InputTypes) -> None:
    _test_stream_roundtrip(algorithm, b'', input_type)


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
@pytest.mark.parametrize('input_type', _input_types)
def test_stream_roundtrip_nul(algorithm: Algorithm,
                              input_type: _InputTypes) -> None:
    _test_stream_roundtrip(algorithm, b'\x00', input_type)


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
@pytest.mark.parametrize('input_type', _input_types)
def test_stream_roundtrip_compressible(algorithm: Algorithm,
                                       input_type: _InputTypes) -> None:
    _test_stream_roundtrip(algorithm,
                           b''.join(c.to_bytes() for c in range(256)) * 4096,
                           input_type)


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
@pytest.mark.parametrize('input_type', _input_types)
def test_stream_roundtrip_random(algorithm: Algorithm,
                                 input_type: _InputTypes) -> None:
    _test_stream_roundtrip(algorithm, random.randbytes(1024 * 1024), input_type)


# Chunked operation doesn’t work with memoryview because of the “TypeError:
# read-only memoryview has nonzero offset” restriction enforced by
# apple_compression.py, due to a ctypes limitation.
@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
@pytest.mark.parametrize('input_type',
                         (input_type if input_type is not memoryview else
                          pytest.param(input_type, marks=pytest.mark.xfail)
                          for input_type in _input_types))
def test_stream_roundtrip_chunks(algorithm: Algorithm,
                                 input_type: _InputTypes) -> None:
    data = b''.join(c.to_bytes() for c in range(256)) * 1024
    data_t = input_type(data) if input_type is not bytes else data

    with Compressor(algorithm) as compressor:
        compressed = b''
        offset = 0
        chunk_size = 0
        while offset < len(data):
            chunk_size = (chunk_size % 16) + 1
            compressed += compressor.compress(data_t[offset:offset +
                                                     chunk_size])
            offset += chunk_size
        compressed += compressor.flush()

    compressed_t = (input_type(compressed)
                    if input_type is not bytes else compressed)

    with Decompressor(algorithm) as decompressor:
        assert not decompressor.eof
        assert decompressor.needs_input
        decompressed = b''
        offset = 0
        chunk_size = 0
        while offset < len(compressed_t):
            assert not decompressor.eof
            assert decompressor.needs_input
            chunk_size = (chunk_size % 16) + 1
            decompressed += decompressor.decompress(compressed_t[offset:offset +
                                                                 chunk_size])
            offset += chunk_size
        decompressed += decompressor.flush()
        assert decompressor.flush() == b''
        assert decompressed == data
        assert decompressor.eof
        assert not decompressor.needs_input


# Chunked operation doesn’t work with memoryview because of the “TypeError:
# read-only memoryview has nonzero offset” restriction enforced by
# apple_compression.py, due to a ctypes limitation.
@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
@pytest.mark.parametrize('input_type',
                         (input_type if input_type is not memoryview else
                          pytest.param(input_type, marks=pytest.mark.xfail)
                          for input_type in _input_types))
def test_stream_roundtrip_decompress_limited(algorithm: Algorithm,
                                             input_type: _InputTypes) -> None:
    data = b''.join(c.to_bytes() for c in range(256)) * 1024
    data_t = input_type(data) if input_type is not bytes else data

    with Compressor(algorithm) as compressor:
        compressed = compressor.compress(data_t)
        compressed += compressor.flush()

    compressed_t = (input_type(compressed)
                    if input_type is not bytes else compressed)

    with Decompressor(algorithm) as decompressor:
        assert not decompressor.eof
        assert decompressor.needs_input
        decompressed = b''
        offset = 0
        compressed_chunk_size = 0
        while offset < len(compressed_t) or not decompressor.eof:
            assert not decompressor.eof
            compressed_chunk_size = (compressed_chunk_size % 16) + 1
            compressed_chunk = compressed_t[offset:offset +
                                            compressed_chunk_size]
            decompressed_chunk_size = ((compressed_chunk_size + 7) % 16)
            decompressed_chunk = decompressor.decompress(
                compressed_chunk, max_length=decompressed_chunk_size)
            assert len(decompressed_chunk) <= decompressed_chunk_size
            decompressed += decompressed_chunk
            offset += compressed_chunk_size
        # decompressor.flush() is intentionally skipped to test that all data is
        # extracted even `flush` is not called.
        assert decompressed == data
        assert decompressor.eof
        assert not decompressor.needs_input


# Chunked operation doesn’t work with memoryview because of the “TypeError:
# read-only memoryview has nonzero offset” restriction enforced by
# apple_compression.py, due to a ctypes limitation.
@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
@pytest.mark.parametrize('input_type',
                         (input_type if input_type is not memoryview else
                          pytest.param(input_type, marks=pytest.mark.xfail)
                          for input_type in _input_types))
def test_stream_roundtrip_decompress_limited_flush(
        algorithm: Algorithm, input_type: _InputTypes) -> None:
    data = b''.join(c.to_bytes() for c in range(256)) * 1024
    data_t = input_type(data) if input_type is not bytes else data

    with Compressor(algorithm) as compressor:
        compressed = compressor.compress(data_t)
        compressed += compressor.flush()

    compressed_t = (input_type(compressed)
                    if input_type is not bytes else compressed)

    with Decompressor(algorithm) as decompressor:
        assert not decompressor.eof
        assert decompressor.needs_input
        decompressed = b''
        offset = 0
        compressed_chunk_size = 0
        while offset < len(compressed_t):
            assert not decompressor.eof
            compressed_chunk_size = (compressed_chunk_size % 16) + 1
            compressed_chunk = compressed_t[offset:offset +
                                            compressed_chunk_size]
            decompressed_chunk_size = ((compressed_chunk_size + 7) % 16)
            decompressed_chunk = decompressor.decompress(
                compressed_chunk, max_length=decompressed_chunk_size)
            assert len(decompressed_chunk) <= decompressed_chunk_size
            decompressed += decompressed_chunk
            offset += compressed_chunk_size
        assert not decompressor.eof
        assert not decompressor.needs_input
        decompressed_chunk = decompressor.flush()
        assert len(decompressed_chunk)
        decompressed += decompressed_chunk
        assert decompressor.flush() == b''
        assert decompressed == data
        assert decompressor.eof
        assert not decompressor.needs_input


@pytest.mark.parametrize('algorithm', _interoperable_algorithms.keys())
def test_stream_decompress_interoperable(algorithm: Algorithm) -> None:
    data = random.randbytes(1024 * 1024)
    interoperable_implementation = _interoperable_algorithms[algorithm]
    compressed = interoperable_implementation.compress(data)
    with Decompressor(algorithm) as decompressor:
        assert not decompressor.eof
        assert decompressor.needs_input
        decompressed = decompressor.decompress(compressed)
        decompressed += decompressor.flush()
        assert decompressor.flush() == b''
        assert decompressed == data
        assert decompressor.eof
        assert not decompressor.needs_input


@pytest.mark.parametrize('algorithm', _interoperable_algorithms.keys())
def test_stream_compress_interoperable(algorithm: Algorithm) -> None:
    data = random.randbytes(1024 * 1024)
    interoperable_implementation = _interoperable_algorithms[algorithm]
    with Compressor(algorithm) as compressor:
        compressed = compressor.compress(data)
        compressed += compressor.flush()
    decompressed = interoperable_implementation.decompress(compressed)
    assert decompressed == data


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
def test_stream_compress_after_flush(algorithm: Algorithm) -> None:
    with Compressor(algorithm) as compressor:
        compressed = compressor.compress(b'01234567')
        compressed += compressor.flush()
        with pytest.raises(ValueError) as exc_info:
            compressed = compressor.compress(b'89abcdef')
        assert exc_info.value.args == ('Compressor has been flushed',)

    with Compressor(algorithm) as compressor:
        compressed = compressor.compress(b'01234567')
        compressed += compressor.flush()
        with pytest.raises(ValueError) as exc_info:
            compressed = compressor.flush()
        assert exc_info.value.args == ('Repeated call to flush()',)


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
def test_stream_decompress_after_eof(algorithm: Algorithm) -> None:
    data = b'01234567'
    compressed = compress(data, algorithm)
    with Decompressor(algorithm) as decompressor:
        decompressed = decompressor.decompress(compressed)
        assert decompressed == data
        with pytest.raises(EOFError) as exc_info:
            decompressor.decompress(b'trailing garbage')
        assert exc_info.value.args == ('Already at end of stream',)


@pytest.mark.parametrize('algorithm',
                         (algorithm if algorithm not in _no_stream_algorithms
                          else pytest.param(algorithm, marks=pytest.mark.skip)
                          for algorithm in Algorithm))
def test_stream_decompress_truncated(algorithm: Algorithm) -> None:
    data = b'0123456789abcdef'
    compressed = compress(data, algorithm)
    with Decompressor(algorithm) as decompressor:
        decompressor.decompress(compressed[:-4])
        assert not decompressor.eof
        assert decompressor.needs_input
        with pytest.raises(CompressionError):
            decompressor.flush()


if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:]))
