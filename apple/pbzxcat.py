#!/usr/bin/env python3

import abc
import argparse
import io
import lzma
import os
import queue
import struct
import sys
import threading
import typing


class FormatError(Exception):
    pass


def _StructReadUnpack(file: typing.BinaryIO,
                      format: str) -> tuple[typing.Any, ...]:
    s = struct.Struct(format)

    data = file.read(s.size)
    if len(data) == 0:
        raise EOFError

    return s.unpack(data)


class _LimitedReaderBase(io.RawIOBase):

    @abc.abstractmethod
    def CheckFinished(self) -> None:
        ...


class _LimitedInputSizeReader(_LimitedReaderBase):

    def __init__(self, source: typing.BinaryIO, size: int):
        self._source = source
        self._size = size

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()

        if size is None or size < 0:
            return self.readall()

        size = min(self._size, size)
        output = self._source.read(size)
        self._size -= len(output)
        return output

    @typing.override
    def CheckFinished(self) -> None:
        if self._size != 0:
            raise FormatError('unconsumed data')


class _ExactOutputSizeReader(_LimitedReaderBase):

    def __init__(self, source: _LimitedReaderBase, size: int):
        self._source = source
        self._size = size
        self._eof = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()

        if size is None or size < 0:
            return self.readall()

        output = self._source.read(size)
        if output == b'':
            self._eof = True
        else:
            self._size -= len(output)

        return output

    @typing.override
    def CheckFinished(self) -> None:
        if self._eof and self._size > 0:
            raise FormatError('unconsumed data')

        if self._size < 0:
            raise FormatError('excess data')

        self._source.CheckFinished()


class LZMAReader(_LimitedReaderBase):

    def __init__(self, source: _LimitedReaderBase | io.BytesIO, *args:
                 typing.Any, **kwargs: typing.Any):
        self._source = source
        self._decompressor = lzma.LZMADecompressor(*args, **kwargs)

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()

        if size is None or size < 0:
            return self.readall()

        if self._decompressor.eof:
            return b''

        output = b''
        while output == b'' and not self._decompressor.eof:
            if self._decompressor.needs_input:
                input = self._source.read(size)
            else:
                input = b''

            output = self._decompressor.decompress(input, size)

        return output

    @typing.override
    def CheckFinished(self) -> None:
        if not self._decompressor.eof:
            raise FormatError('unterminated LZMA stream')
        if self._decompressor.unused_data != b'':
            raise FormatError('unconsumed data following LZMA stream')

        # self._source may be io.BytesIO, which does not implement
        # CheckFinished.
        if isinstance(self._source, _LimitedReaderBase):
            self._source.CheckFinished()


class _PBZXBlockSegmenter:

    def __init__(self, file: typing.BinaryIO, decompress: bool = True):
        self._file = file
        self._decompress = decompress
        self._block_decompressed_size = None

        (magic, self._base_block_decompressed_size) = _StructReadUnpack(
            self._file, '>4sQ')
        if magic != b'pbzx':
            raise FormatError('not pbzx')

    def __iter__(self) -> typing.Iterator[tuple[_LimitedReaderBase, bool, int]]:
        return self

    def __next__(self) -> tuple[_LimitedReaderBase, bool, int]:
        old_block_decompressed_size = self._block_decompressed_size
        try:
            (self._block_decompressed_size,
             block_compressed_size) = _StructReadUnpack(self._file, '>QQ')
            assert isinstance(self._block_decompressed_size, int)
            assert isinstance(block_compressed_size, int)
        except EOFError:
            raise StopIteration

        # The size of all blocks must match the base block size, except the
        # final block is permitted to be shorter. In no case may a block be
        # longer than the base block size.
        if (self._block_decompressed_size > self._base_block_decompressed_size):
            raise FormatError('block size exceeds base block size')

        # Since there's a new block, the previous block, if any, was not the
        # final block.
        if (old_block_decompressed_size is not None and
            (old_block_decompressed_size
             != self._base_block_decompressed_size)):
            raise FormatError(
                'non-final block size does not match base block size')

        if block_compressed_size > self._block_decompressed_size:
            raise FormatError('compressed data too large')

        compressed = block_compressed_size < self._block_decompressed_size
        limited_size_reader = _LimitedInputSizeReader(self._file,
                                                      block_compressed_size)
        if compressed and self._decompress:
            block_reader: _LimitedReaderBase = LZMAReader(limited_size_reader)
        else:
            block_reader = limited_size_reader

        return (block_reader, compressed, self._block_decompressed_size)


class PBZXReader(io.RawIOBase):

    def __init__(self, source: typing.BinaryIO):
        self._pbzx_blocks = iter(_PBZXBlockSegmenter(source))
        self._eof = False
        self._block_reader: _ExactOutputSizeReader | None = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()

        if size is None or size < 0:
            return self.readall()

        output = b''
        while output == b'' and not self._eof:
            if self._block_reader is None:
                try:
                    (block_reader, compressed,
                     block_decompressed_size) = next(self._pbzx_blocks)
                    self._block_reader = _ExactOutputSizeReader(
                        block_reader, block_decompressed_size)
                except StopIteration:
                    self._eof = True
                    break

            output = self._block_reader.read(size)
            if output == b'':
                self._block_reader.CheckFinished()
                self._block_reader = None

        return output


def _PBZXDecompressThread(
    input_queue: queue.Queue[tuple[int, _ExactOutputSizeReader] | None],
    output_queue: queue.Queue[tuple[int, bytes]],
) -> None:
    while True:
        item = input_queue.get()
        if item is None:
            input_queue.task_done()
            break

        (block_sequence, compressed_block_reader) = item
        decompressed_block = compressed_block_reader.read()
        compressed_block_reader.CheckFinished()

        output_queue.put((block_sequence, decompressed_block))

        input_queue.task_done()


def PBZXCat(in_file: typing.BinaryIO,
            out_file: typing.BinaryIO,
            max_threads: int | None = None) -> None:
    if max_threads == 1:
        lzma_reader = PBZXReader(in_file)
        while True:
            out = lzma_reader.read(16 * 1024)
            if out == b'':
                break
            out_file.write(out)

        return

    if max_threads is None:
        try:
            max_threads = len(
                os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except AttributeError:
            max_threads = os.cpu_count()
        assert max_threads is not None

    pbzx_blocks = iter(_PBZXBlockSegmenter(in_file, False))

    # TODO: the whole rest of this function can be tightened up.
    input_queue: queue.Queue[tuple[int, _ExactOutputSizeReader] |
                             None] = (queue.Queue(max_threads))
    output_queue: queue.Queue[tuple[int, bytes]] = queue.Queue()

    threads: list[threading.Thread] = []

    pending_input = None
    input_block_sequence = 0
    pending_output: list[bytes | None] = []
    output_block_sequence = 0
    output_eof = False
    input_eof = False
    while not output_eof:
        if not input_eof and pending_input is None:
            try:
                (block_reader, compressed,
                 block_decompressed_size) = next(pbzx_blocks)

                if compressed:
                    compressed_block_reader = _ExactOutputSizeReader(
                        LZMAReader(io.BytesIO(block_reader.read())),
                        block_decompressed_size)
                    pending_input = (input_block_sequence,
                                     compressed_block_reader)
                    pending_output.append(None)
                else:
                    decompressed_block_reader = _ExactOutputSizeReader(
                        block_reader, block_decompressed_size)
                    decompressed_block = decompressed_block_reader.read()
                    decompressed_block_reader.CheckFinished()
                    pending_output.append(decompressed_block)

                input_block_sequence += 1
            except StopIteration:
                input_eof = True

        input_queue_full = False
        if pending_input is not None:
            try:
                input_queue.put(pending_input, block=input_eof)
                pending_input = None

                if input_queue.qsize() != 0 and len(threads) < max_threads:
                    thread = threading.Thread(target=_PBZXDecompressThread,
                                              args=(input_queue, output_queue),
                                              daemon=True)
                    thread.start()
                    threads.append(thread)
            except queue.Full:
                input_queue_full = True

        try:
            (decompressed_block_sequence,
             decompressed_block) = output_queue.get(
                 block=input_eof or input_queue_full)
            pending_output[decompressed_block_sequence] = decompressed_block
        except queue.Empty:
            pass

        while output_block_sequence < len(pending_output) and pending_output[
                output_block_sequence] is not None:
            out_file.write(
                pending_output[output_block_sequence])  # type: ignore[arg-type]
            pending_output[output_block_sequence] = None
            output_block_sequence += 1
            if input_eof and output_block_sequence == input_block_sequence:
                output_eof = True

    input_queue.join()

    for thread in threads:
        input_queue.put(None)

    for thread in threads:
        thread.join()


def main(args: list[str]) -> int | None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('file', nargs='*')
    parsed = parser.parse_args(args)

    if len(parsed.file) == 0:
        if not parsed.force and sys.stdin.isatty():
            print('%s: standard input is a terminal -- ignoring' %
                  os.path.basename(sys.argv[0]))
            return 1
        PBZXCat(sys.stdin.buffer, sys.stdout.buffer)
    else:
        for path in parsed.file:
            with open(path, 'rb') as file:
                PBZXCat(file, sys.stdout.buffer)

    return None


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
