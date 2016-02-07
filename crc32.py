#!/usr/bin/env python
# CRC32 tools by Victor

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# References
# Calculating Reverse CRC http://www.danielvik.com/2010/10/calculating-reverse-crc.html
# Finding Reverse CRC Patch with Readable Characters http://www.danielvik.com/2012/01/finding-reverse-crc-patch-with-readable.html
# Rewinding CRC - Calculating CRC backwards
# http://www.danielvik.com/2013/07/rewinding-crc-calculating-crc-backwards.html

import argparse
import os
import sys

permitted_characters = set(
    map(ord, 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890_'))  # \w

testing = False

args = None


def get_poly():
    poly = parse_dword(args.poly)
    if args.msb:
        poly = reverseBits(poly)
    check32(poly)
    return poly


def get_input():
    if args.instr:
        return tuple(map(ord, args.instr))
    with args.infile as f:  # pragma: no cover
        return tuple(map(ord, f.read()))


def out(msg):
    if not testing:  # pragma: no cover
        args.outfile.write(msg)
        args.outfile.write(os.linesep)

table = []
table_reverse = []


def init_tables(poly, reverse=True):
    global table, table_reverse
    table = []
    # build CRC32 table
    for i in range(256):
        for j in range(8):
            if i & 1:
                i >>= 1
                i ^= poly
            else:
                i >>= 1
        table.append(i)
    assert len(table) == 256, "table is wrong size"
    # build reverse table
    if reverse:
        table_reverse = []
        found_none = set()
        found_multiple = set()
        for i in range(256):
            found = []
            for j in range(256):
                if table[j] >> 24 == i:
                    found.append(j)
            table_reverse.append(tuple(found))
            if not found:
                found_none.add(i)
            elif len(found) > 1:
                found_multiple.add(i)
        assert len(table_reverse) == 256, "reverse table is wrong size"
        if found_multiple:
            out('WARNING: Multiple table entries have an MSB in {0}'.format(
                rangess(found_multiple)))
        if found_none:
            out('ERROR: no MSB in the table equals bytes in {0}'.format(
                rangess(found_none)))


def calc(data, accum=0):
    accum = ~accum
    for b in data:
        accum = table[(accum ^ b) & 0xFF] ^ ((accum >> 8) & 0x00FFFFFF)
    accum = ~accum
    return accum & 0xFFFFFFFF


def rewind(accum, data):
    if not data:
        return (accum,)
    stack = [(len(data), ~accum)]
    solutions = set()
    while stack:
        node = stack.pop()
        prev_offset = node[0] - 1
        for i in table_reverse[(node[1] >> 24) & 0xFF]:
            prevCRC = (((node[1] ^ table[i]) << 8) |
                       (i ^ data[prev_offset])) & 0xFFFFFFFF
            if prev_offset:
                stack.append((prev_offset, prevCRC))
            else:
                solutions.add((~prevCRC) & 0xFFFFFFFF)
    return solutions


def findReverse(desired, accum):
    solutions = set()
    accum = ~accum
    stack = [(~desired,)]
    while stack:
        node = stack.pop()
        for j in table_reverse[(node[0] >> 24) & 0xFF]:
            if len(node) == 4:
                a = accum
                data = []
                node = node[1:] + (j,)
                for i in range(3, -1, -1):
                    data.append((a ^ node[i]) & 0xFF)
                    a >>= 8
                    a ^= table[node[i]]
                solutions.add(tuple(data))
            else:
                stack.append(((node[0] ^ table[j]) << 8,) + node[1:] + (j,))
    return solutions

# Tools


def parse_dword(x):
    return int(x, 0) & 0xFFFFFFFF


def reverseBits(x):
    # http://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
    # http://stackoverflow.com/a/20918545
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1)
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2)
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4)
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8)
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16)
    return x & 0xFFFFFFFF

# Compatibility with Python 2.6 and earlier.
if hasattr(int, "bit_length"):
    def bit_length(num):
        return num.bit_length()
else:
    def bit_length(n):
        if n == 0:
            return 0
        bits = -32
        m = 0
        while n:
            m = n
            n >>= 32
            bits += 32
        while m:
            m >>= 1
            bits += 1
        return bits


def check32(poly):
    if poly & 0x80000000 == 0:
        out('WARNING: polynomial degree ({0}) != 32'.format(bit_length(poly)))
        out('         instead, try')
        out('         0x{0:08x} (reversed/lsbit-first)'.format(poly | 0x80000000))
        out('         0x{0:08x} (normal/msbit-first)'.format(reverseBits(poly | 0x80000000)))


def reciprocal(poly):
    ''' Return the reversed reciprocal (Koopman notatation) polynomial of a
        reversed (lsbit-first) polynomial '''
    return reverseBits((poly << 1) | 1)


def print_num(num):
    ''' Write a numeric result in various forms '''
    out('hex: 0x{0:08x}'.format(num))
    out('dec:   {0:d}'.format(num))
    out('oct: 0o{0:011o}'.format(num))
    out('bin: 0b{0:032b}'.format(num))

import itertools


def ranges(i):
    for kg in itertools.groupby(enumerate(i), lambda x: x[1] - x[0]):
        g = list(kg[1])
        yield g[0][1], g[-1][1]


def rangess(i):
    return ', '.join(map(lambda x: '[{0},{1}]'.format(*x), ranges(i)))

# Parsers


def get_parser():
    ''' Return the command-line parser '''
    parser = argparse.ArgumentParser(
        description="Reverse, undo, and calculate CRC32 checksums")
    subparsers = parser.add_subparsers(metavar='action')

    poly_flip_parser = argparse.ArgumentParser(add_help=False)
    subparser_group = poly_flip_parser.add_mutually_exclusive_group()
    subparser_group.add_argument(
        '-m', '--msbit', dest="msb", action='store_true',
        help='treat the polynomial as normal (msbit-first)')
    subparser_group.add_argument('-l', '--lsbit', action='store_false',
                                 help='treat the polynomial as reversed (lsbit-first) [default]')

    desired_poly_parser = argparse.ArgumentParser(add_help=False)
    desired_poly_parser.add_argument(
        'desired', type=str, help='[int] desired checksum')

    default_poly_parser = argparse.ArgumentParser(add_help=False)
    default_poly_parser.add_argument(
        'poly', default='0xEDB88320', type=str, nargs='?',
        help='[int] polynomial [default: 0xEDB88320]')

    accum_parser = argparse.ArgumentParser(add_help=False)
    accum_parser.add_argument(
        'accum', type=str, help='[int] accumulator (final checksum)')

    default_accum_parser = argparse.ArgumentParser(add_help=False)
    default_accum_parser.add_argument(
        'accum', default='0', type=str, nargs='?',
        help='[int] starting accumulator [default: 0]')

    outfile_parser = argparse.ArgumentParser(add_help=False)
    outfile_parser.add_argument('-o', '--outfile',
                                metavar="f",
                                type=argparse.FileType('w'),
                                default=sys.stdout,
                                help="Output to a file instead of stdout")

    infile_parser = argparse.ArgumentParser(add_help=False)
    subparser_group = infile_parser.add_mutually_exclusive_group()
    subparser_group.add_argument('-i', '--infile',
                                 metavar="f",
                                 type=argparse.FileType('rb'),
                                 default=sys.stdin,
                                 help="Input from a file instead of stdin")
    subparser_group.add_argument('-s', '--str',
                                 metavar="s",
                                 type=str,
                                 default='',
                                 dest='instr',
                                 help="Use a string as input")

    subparser = subparsers.add_parser('flip', parents=[outfile_parser],
                                      help="flip the bits to convert normal(msbit-first) polynomials to reversed (lsbit-first) and vice versa")
    subparser.add_argument('poly', type=str, help='[int] polynomial')
    subparser.set_defaults(
        func=lambda: print_num(reverseBits(parse_dword(args.poly))))

    subparser = subparsers.add_parser('reciprocal', parents=[outfile_parser],
                                      help="find the reciprocal (Koopman notation) of a reversed (lsbit-first) polynomial and vice versa")
    subparser.add_argument('poly', type=str, help='[int] polynomial')
    subparser.set_defaults(func=reciprocal_callback)

    subparser = subparsers.add_parser('table', parents=[outfile_parser,
                                                        poly_flip_parser,
                                                        default_poly_parser],
                                      help="generate a lookup table for a polynomial")
    subparser.set_defaults(func=table_callback)

    subparser = subparsers.add_parser('reverse', parents=[
        outfile_parser,
        poly_flip_parser,
        desired_poly_parser,
        default_accum_parser,
        default_poly_parser],
        help="find a patch that causes the CRC32 checksum to become a desired value")
    subparser.set_defaults(func=reverse_callback)

    subparser = subparsers.add_parser('undo', parents=[
        outfile_parser,
        poly_flip_parser,
        accum_parser,
        default_poly_parser,
        infile_parser],
        help="rewind a CRC32 checksum")
    subparser.add_argument('-n', '--len', metavar='l', type=str,
                           default='0', help='[int] number of bytes to rewind [default: 0]')
    subparser.set_defaults(func=undo_callback)

    subparser = subparsers.add_parser('calc', parents=[
        outfile_parser,
        poly_flip_parser,
        default_accum_parser,
        default_poly_parser,
        infile_parser],
        help="calculate the CRC32 checksum")
    subparser.set_defaults(func=calc_callback)

    return parser


def reciprocal_callback():
    poly = parse_dword(args.poly)
    check32(poly)
    print_num(reciprocal(poly))


def table_callback():
    # initialize tables
    init_tables(get_poly(), False)
    # print table
    out('[{0}]'.format(', '.join(map('0x{0:08x}'.format, table))))


def reverse_callback():
    # initialize tables
    init_tables(get_poly())
    # find reverse bytes
    desired = parse_dword(args.desired)
    accum = parse_dword(args.accum)
    # 4-byte patch
    patches = findReverse(desired, accum)
    for patch in patches:
        out('4 bytes: {{0x{0:02x}, 0x{1:02x}, 0x{2:02x}, 0x{3:02x}}}'.format(*patch))
        checksum = calc(patch, accum)
        out('verification checksum: 0x{0:08x} ({1})'.format(
            checksum, 'OK' if checksum == desired else 'ERROR'))
    # 6-byte alphanumeric patches
    for i in permitted_characters:
        for j in permitted_characters:
            patch = [i, j]
            patches = findReverse(desired, calc(patch, accum))
            for last_4_bytes in patches:
                if all(p in permitted_characters for p in last_4_bytes):
                    patch.extend(last_4_bytes)
                    out('alternative: {1}{2}{3}{4}{5}{6} ({0})'.format(
                        'OK' if calc(patch, accum) == desired else 'ERROR', *map(chr, patch)))


def undo_callback():
    # initialize tables
    init_tables(get_poly())
    # calculate checksum
    accum = parse_dword(args.accum)
    maxlen = int(args.len, 0)
    data = get_input()
    if not 0 < maxlen <= len(data):
        maxlen = len(data)
    out('rewinded {0}/{1} ({2:.2f}%)'.format(maxlen, len(data),
        maxlen * 100.0 / len(data) if len(data) else 100))
    for solution in rewind(accum, data[-maxlen:]):
        out('')
        print_num(solution)


def calc_callback():
    # initialize tables
    init_tables(get_poly(), False)
    # calculate checksum
    accum = parse_dword(args.accum)
    data = get_input()
    out('data len: {0}'.format(len(data)))
    out('')
    print_num(calc(data, accum))


def main(argv=None):
    ''' Runs the program and handles command line options '''
    parser = get_parser()

    # Parse arguments and run the function
    global args
    args = parser.parse_args(argv)
    args.func()

if __name__ == '__main__':
    main()  # pragma: no cover
