#!/usr/bin/env python3

# pyright: strict

from __future__ import annotations

import abc
import argparse
import ctypes
import enum
import sys
import types
import typing


class CompressionError(Exception):
    pass


# compression_algorithm
class Algorithm(enum.Enum):
    LZ4 = 0x0100  # OS X 10.11
    ZLIB = 0x0205  # OS X 10.11
    LZMA = 0x0306  # OS X 10.11
    LZ4_RAW = 0x0101  # OS X 10.11, buffer API only
    BROTLI = 0x0b02  # macOS 12
    LZFSE = 0x0801  # OS X 10.11
    LZBITMAP = 0x0702  # macOS 12, buffer API only


# compression_stream
class _CompressionStream(ctypes.Structure):
    _fields_ = (
        ('dst_ptr', ctypes.c_char_p),
        ('dst_size', ctypes.c_size_t),
        ('src_ptr', ctypes.c_char_p),
        ('src_size', ctypes.c_size_t),
        ('state', ctypes.c_void_p),
    )


# compression_stream_operation
class _StreamOperation(enum.Enum):
    ENCODE = 0  # compress
    DECODE = 1  # decompress


# compression_stream_flags
class _StreamFlags(enum.Flag):
    NONE = 0x0000
    FINALIZE = 0x0001


# compression_status
class _Status(enum.Enum):
    OK = 0
    ERROR = -1
    END = 1


def _buf_to_const_char_p_and_size(
        buf: bytes | bytearray | memoryview) -> tuple[ctypes.c_char_p, int]:
    if isinstance(buf, bytearray):
        return _buf_to_mutable_char_p_and_size(buf)

    if isinstance(buf, memoryview):
        if not buf.readonly:
            return _buf_to_mutable_char_p_and_size(buf)

        # ctypes doesn’t provide a direct way to get a pointer to read-only
        # memoryview. `(ctypes.c_char * size).from_buffer(buf)` raises
        # “TypeError: underlying buffer is not writable”. `ctypes.cast(buf,
        # ctypes.c_char_p)` (used for bytes objects below) raises
        # “ctypes.ArgumentError: argument 1: TypeError: 'memoryview' object
        # cannot be interpreted as ctypes.c_void_p”.
        #
        # As a workaround, access the memoryview’s underlying object directly.
        #
        # Perform the same checking that _CDataType.from_buffer does (Python
        # Modules/_ctypes/_ctypes.c CDataType_from_buffer_impl). Namely, the
        # buffer must be C-contiguous.
        if not buf.c_contiguous:
            raise TypeError('underlying buffer is not C contiguous')

        assert isinstance(buf.obj, bytes)

        # Unfortunately, there’s no way to determine the memoryview’s offset
        # within the underlying buffer (2 in `memoryview(b'abcdef')[2:]`), so
        # this can only work when the memoryview’s size is the same as the
        # underlying object’s size, meaning that the offset must be 0.
        if len(buf) != len(buf.obj):
            raise TypeError('read-only memoryview has nonzero offset')

        buf = buf.obj

    # ctypes doesn’t provide a direct way to get a pointer to a bytes object
    # either. `(ctypes.c_char * size).from_buffer(buf)` raises “TypeError:
    # underlying buffer is not writable”. As a workaround, cast the buffer to
    # ctypes.c_char_p. https://github.com/python/cpython/issues/55636.
    buf_cp = ctypes.cast(
        buf,  # type: ignore[arg-type]
        ctypes.c_char_p)
    return (buf_cp, len(buf))


def _buf_to_mutable_char_p_and_size(
        buf: bytearray | memoryview) -> tuple[ctypes.c_char_p, int]:
    size = len(buf)
    return (ctypes.cast((ctypes.c_char * size).from_buffer(buf),
                        ctypes.c_char_p), size)


class _Compression:
    __slots__ = (
        '_lib',
        '_encode_scratch_buffer_size',
        '_encode_buffer',
        '_decode_scratch_buffer_size',
        '_decode_buffer',
        '_stream_init',
        '_stream_process',
        '_stream_destroy',
    )

    def __init__(self) -> None:
        self._lib = ctypes.CDLL('/usr/lib/libcompression.dylib')

        self._encode_scratch_buffer_size = ctypes.CFUNCTYPE(
            ctypes.c_size_t,
            ctypes.c_int,
        )(
            ('compression_encode_scratch_buffer_size', self._lib),
            ((1, 'algorithm'),),
        )

        self._encode_buffer = ctypes.CFUNCTYPE(
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_int,
        )(
            ('compression_encode_buffer', self._lib),
            (
                (1, 'dst_buffer'),
                (1, 'dst_size'),
                (1, 'src_buffer'),
                (1, 'src_size'),
                (1, 'scratch_buffer'),
                (1, 'algorithm'),
            ),
        )

        self._decode_scratch_buffer_size = ctypes.CFUNCTYPE(
            ctypes.c_size_t,
            ctypes.c_int,
        )(
            ('compression_decode_scratch_buffer_size', self._lib),
            ((1, 'algorithm'),),
        )

        self._decode_buffer = ctypes.CFUNCTYPE(
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_int,
        )(
            ('compression_decode_buffer', self._lib),
            (
                (1, 'dst_buffer'),
                (1, 'dst_size'),
                (1, 'src_buffer'),
                (1, 'src_size'),
                (1, 'scratch_buffer'),
                (1, 'algorithm'),
            ),
        )

        self._stream_init = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        )(
            ('compression_stream_init', self._lib),
            (
                (1, 'stream'),
                (1, 'operation'),
                (1, 'algorithm'),
            ),
        )

        self._stream_process = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        )(
            ('compression_stream_process', self._lib),
            (
                (1, 'stream'),
                (1, 'flags'),
            ),
        )

        self._stream_destroy = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
        )(
            ('compression_stream_destroy', self._lib),
            ((1, 'stream'),),
        )

    def encode_scratch_buffer_size(self, algorithm: Algorithm) -> int:
        result = self._encode_scratch_buffer_size(algorithm.value)
        assert isinstance(result, int)
        return result

    def encode_buffer(
            self,
            dst: bytearray | memoryview,
            src: bytes | bytearray | memoryview,
            algorithm: Algorithm,
            scratch_buffer: bytearray | memoryview | None = None) -> memoryview:
        if scratch_buffer is not None:
            assert (len(scratch_buffer)
                    >= self.encode_scratch_buffer_size(algorithm))

        result = self._encode_buffer(
            *_buf_to_mutable_char_p_and_size(dst),
            *_buf_to_const_char_p_and_size(src),
            _buf_to_mutable_char_p_and_size(scratch_buffer)[0]
            if scratch_buffer is not None else None, algorithm.value)
        if result == 0:
            # It’s not possible to disambiguate between an error and a
            # compression that validly produced no data. Most algorithms will
            # output a small amount of data even when compressing empty input,
            # but LZ4_RAW does not. Conservatively assume error.
            raise CompressionError

        dst_mv = dst if isinstance(dst, memoryview) else memoryview(dst)
        return dst_mv[:result]

    def decode_scratch_buffer_size(self, algorithm: Algorithm) -> int:
        result = self._decode_scratch_buffer_size(algorithm.value)
        assert isinstance(result, int)
        return result

    def decode_buffer(
            self,
            dst: bytearray | memoryview,
            src: bytes | bytearray | memoryview,
            algorithm: Algorithm,
            scratch_buffer: bytearray | memoryview | None = None) -> memoryview:
        if scratch_buffer is not None:
            assert (len(scratch_buffer)
                    >= self.decode_scratch_buffer_size(algorithm))

        result = self._decode_buffer(
            *_buf_to_mutable_char_p_and_size(dst),
            *_buf_to_const_char_p_and_size(src),
            _buf_to_mutable_char_p_and_size(scratch_buffer)[0]
            if scratch_buffer is not None else None, algorithm.value)
        if result == 0:
            # It’s not possible to disambiguate between an error and a
            # decompression that validly produced no data, which would be the
            # result of decompress(compress(empty)). Conservatively assume
            # error.
            raise CompressionError

        dst_mv = dst if isinstance(dst, memoryview) else memoryview(dst)
        return dst_mv[:result]

    def stream_init(self, stream: _CompressionStream,
                    operation: _StreamOperation, algorithm: Algorithm) -> None:
        status = _Status(
            self._stream_init(ctypes.pointer(stream), operation.value,
                              algorithm.value))
        if status == _Status.ERROR:
            raise CompressionError
        assert status == _Status.OK

    def stream_process(self,
                       stream: _CompressionStream,
                       flags: _StreamFlags = _StreamFlags.NONE) -> _Status:
        status = _Status(
            self._stream_process(ctypes.pointer(stream), flags.value))
        if status == _Status.ERROR:
            raise CompressionError
        return status

    def stream_destroy(self, stream: _CompressionStream) -> None:
        status = _Status(self._stream_destroy(ctypes.pointer(stream)))
        if status == _Status.ERROR:
            raise CompressionError
        assert status == _Status.OK


_compression = _Compression()


def compress_scratch_buffer_size(algorithm: Algorithm) -> int:
    return _compression.encode_scratch_buffer_size(algorithm)


def _compress_buffer_internal(
        decompressed: bytes | bytearray | memoryview,
        algorithm: Algorithm,
        size: int,
        scratch_buffer: bytearray | memoryview | None = None) -> bytes:
    buffer = bytearray(size)
    return bytes(
        _compression.encode_buffer(buffer, decompressed, algorithm,
                                   scratch_buffer))


def compress(decompressed: bytes | bytearray | memoryview,
             algorithm: Algorithm,
             *,
             max_size: int | None = None,
             scratch_buffer: bytearray | memoryview | None = None) -> bytes:
    if max_size is not None:
        return _compress_buffer_internal(decompressed, algorithm, max_size,
                                         scratch_buffer)

    try:
        return _compress_buffer_internal(decompressed, algorithm,
                                         max(int(len(decompressed) * 1.1), 64),
                                         scratch_buffer)
    except CompressionError:
        return _compress_buffer_internal(decompressed, algorithm,
                                         max(len(decompressed) * 2, 128),
                                         scratch_buffer)


def decompress_scratch_buffer_size(algorithm: Algorithm) -> int:
    return _compression.decode_scratch_buffer_size(algorithm)


def _decompress_buffer_internal(
        compressed: bytes | bytearray | memoryview,
        algorithm: Algorithm,
        size: int,
        scratch_buffer: bytearray | memoryview | None = None) -> bytes:
    buffer = bytearray(size)
    return bytes(
        _compression.decode_buffer(buffer, compressed, algorithm,
                                   scratch_buffer))


def decompress(compressed: bytes | bytearray | memoryview,
               algorithm: Algorithm,
               *,
               max_size: int | None = None,
               scratch_buffer: bytearray | memoryview | None = None) -> bytes:
    if max_size is not None:
        decompressed = _decompress_buffer_internal(compressed, algorithm,
                                                   max_size + 1, scratch_buffer)
        if len(decompressed) > max_size:
            raise CompressionError

        return decompressed

    size = len(compressed)
    while True:
        size *= 2
        decompressed = _decompress_buffer_internal(compressed, algorithm,
                                                   size + 1, scratch_buffer)
        if len(decompressed) <= size:
            return decompressed


class _CompressDecompressStream(abc.ABC):
    __slots__ = (
        '_stream',
        '_input_eof',
        '_stream_eof',
    )

    def __init__(self, operation: _StreamOperation, algorithm: Algorithm):
        self._stream: _CompressionStream | None = _CompressionStream()
        self._input_eof = False
        self._stream_eof = False

        _compression.stream_init(self._stream, operation, algorithm)

    def __del__(self) -> None:
        self._close()

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        self._close()
        return None

    def _close(self) -> None:
        if self._stream is not None:
            _compression.stream_destroy(self._stream)
            self._stream = None

    def _process(self,
                 data: bytes | bytearray | memoryview,
                 *,
                 eof: bool = False,
                 output_buffer_size: int | None = None) -> bytes:
        assert not self._input_eof
        if self._stream_eof:
            raise EOFError('Already at end of stream')

        assert self._stream is not None
        self._input_eof = eof

        (self._stream.src_ptr,
         self._stream.src_size) = _buf_to_const_char_p_and_size(data)

        result_ba = bytearray()
        output_buffer = bytearray(output_buffer_size if output_buffer_size
                                  is not None else max(2 * len(data), 64 *
                                                       1024))
        status = _Status.ERROR
        while not self._stream_eof and (self._stream.src_size != 0 or eof):
            (
                self._stream.dst_ptr,
                self._stream.dst_size,
            ) = _buf_to_mutable_char_p_and_size(output_buffer)

            status = _compression.stream_process(
                self._stream,
                _StreamFlags.FINALIZE if eof else _StreamFlags.NONE)
            if status == _Status.ERROR:
                raise CompressionError

            if status == _Status.END:
                self._stream_eof = True

            process_output_produced = memoryview(output_buffer)
            if self._stream.dst_size != 0:
                process_output_produced = (
                    process_output_produced[:-self._stream.dst_size])

            result_ba.extend(process_output_produced)

        # In practice, compression_stream_process reports consuming the entire
        # buffer, even when decompressing and there is data beyond the
        # end-of-stream marker. (It doesn’t actually decompress anything beyond
        # the end of stream.) But this means that there’s no point in trying to
        # keep track of unused_data.

        if self._input_eof or self._stream_eof:
            self._close()

        return bytes(result_ba)

    @abc.abstractmethod
    def process(self, data: bytes) -> bytes:
        ...

    @abc.abstractmethod
    def flush(self) -> bytes:
        ...


class Compressor(_CompressDecompressStream):

    def __init__(self, algorithm: Algorithm):
        super().__init__(_StreamOperation.ENCODE, algorithm)

    def compress(self, data: bytes) -> bytes:
        if self._input_eof:
            raise ValueError('Compressor has been flushed')

        return self._process(data)

    @typing.override
    def process(self, data: bytes) -> bytes:
        return self.compress(data)

    @typing.override
    def flush(self) -> bytes:
        if self._input_eof:
            raise ValueError('Repeated call to flush()')

        return self._process(b'', eof=True)


class Decompressor(_CompressDecompressStream):
    __slots__ = ('_decompressed')

    def __init__(self, algorithm: Algorithm):
        super().__init__(_StreamOperation.DECODE, algorithm)
        self._decompressed = bytearray()

    def decompress(self, data: bytes, max_length: int = -1) -> bytes:
        if self._input_eof:
            raise ValueError('Decompressor has been flushed')

        decompressed = self._process(data)
        if (len(self._decompressed) == 0 and
            (max_length < 0 or max_length >= len(decompressed))):
            result = decompressed
        else:
            self._decompressed.extend(decompressed)

            if max_length < 0 or max_length >= len(self._decompressed):
                result = bytes(self._decompressed)
                self._decompressed.clear()
            else:
                result = bytes(self._decompressed[:max_length])
                self._decompressed = self._decompressed[max_length:]

        return result

    @typing.override
    def process(self, data: bytes) -> bytes:
        return self.decompress(data)

    @typing.override
    def flush(self) -> bytes:
        if self._stream_eof:
            result = bytes(self._decompressed)
            self._decompressed.clear()
            return result

        return self._process(b'', eof=True)

    @property
    def eof(self) -> bool:
        return self._stream_eof and len(self._decompressed) == 0

    @property
    def needs_input(self) -> bool:
        return not self._stream_eof and len(self._decompressed) == 0


class _Interface(enum.Enum):
    BUFFER = 0
    STREAM = 1


def main(args: typing.Sequence[str]) -> int | None:
    parser = argparse.ArgumentParser(
        description='Compress and decompress data using Apple libcompression.')
    parser.add_argument('--mode',
                        choices=('compress', 'decompress'),
                        required=True,
                        help='operation mode')
    parser.add_argument('--algorithm',
                        choices=(
                            'lz4',
                            'zlib',
                            'lzma',
                            'lz4-raw',
                            'brotli',
                            'lzfse',
                            'lzbitmap',
                        ),
                        required=True,
                        help='compression algorithm')
    parser.add_argument(
        '--interface',
        choices=('buffer', 'stream'),
        help='override automatically chosen libcompression interface')
    parser.add_argument('--input',
                        required=False,
                        help='input file, if not stdin')
    parser.add_argument('--output',
                        required=False,
                        help='output file, if not stdout')
    parsed = parser.parse_args(args)

    (compress_decompress_function, compressor_decompressor_class) = {
        'compress': (compress, Compressor),
        'decompress': (decompress, Decompressor),
    }[parsed.mode]

    algorithm = {
        'lz4': Algorithm.LZ4,
        'zlib': Algorithm.ZLIB,
        'lzma': Algorithm.LZMA,
        'lz4-raw': Algorithm.LZ4_RAW,
        'brotli': Algorithm.BROTLI,
        'lzfse': Algorithm.LZFSE,
        'lzbitmap': Algorithm.LZBITMAP,
    }[parsed.algorithm]

    if parsed.interface is None:
        # LZ4_RAW and LZBITMAP are only implemented in the buffer interface, not
        # the stream interface.
        interface = (_Interface.BUFFER if algorithm in (
            Algorithm.LZ4_RAW,
            Algorithm.LZBITMAP,
        ) else _Interface.STREAM)
    else:
        interface = {
            'buffer': _Interface.BUFFER,
            'stream': _Interface.STREAM,
        }[parsed.interface]

    with (
        (sys.stdin.buffer if parsed.input is None else open(parsed.input, 'rb'))
            as input_file,
        (sys.stdout.buffer if parsed.output is None else open(
            parsed.output, 'wb')) as output_file,
    ):
        if interface == _Interface.BUFFER:
            input = input_file.read()
            output = compress_decompress_function(input, algorithm)
            output_file.write(output)
        else:
            assert interface == _Interface.STREAM
            with compressor_decompressor_class(algorithm) as stream:
                eof = False
                while not eof:
                    input = input_file.read(64 * 1024)

                    if len(input) != 0:
                        output = stream.process(input)
                    else:
                        eof = True
                        output = stream.flush()

                    output_file.write(output)

    return None


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
