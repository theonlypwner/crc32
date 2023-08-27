#!/usr/bin/env python
# CRC32 tools by Victor

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
    if args.reciprocal:
        poly = reverseBits(reciprocal(poly))
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
            i = (i >> 1) ^ (poly & -(i & 1))
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


def check32(poly):
    if poly & 0x80000000 == 0:
        suggested = poly | 0x80000000
        out('WARNING: polynomial degree ({0}) != 32'.format(poly.bit_length()))
        out('         instead, try')
        out('         0x{0:08x} (reversed/lsbit-first)'.format(suggested))
        out('         0x{0:08x} (normal/msbit-first)'.format(reverseBits(suggested)))


def reciprocal(poly):
    ''' Return the reciprocal polynomial of a reversed (lsbit-first) polynomial. '''
    return poly << 1 & 0xffffffff | 1


def out_num(num):
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

    desired_poly_parser = argparse.ArgumentParser(add_help=False)
    desired_poly_parser.add_argument(
        'desired', type=str, help='[int] desired checksum')

    default_poly_parser = argparse.ArgumentParser(add_help=False)
    default_poly_parser.add_argument(
        'poly', default='0xEDB88320', type=str, nargs='?',
        help='[int] polynomial [default: 0xEDB88320]')
    subparser_group = default_poly_parser.add_mutually_exclusive_group()
    subparser_group.add_argument(
        '-m', '--msbit', '--normal', dest='msb', action='store_true',
        help='treat the polynomial as normal (msbit-first)')
    subparser_group.add_argument(
        '-l', '--lsbit', '--reversed', action='store_false',
        help='treat the polynomial as reversed (lsbit-first) [default]')
    default_poly_parser.add_argument(
        '-r', '--reciprocal', action='store_true',
        help='treat the polynomial as reciprocal (Koopman notation is reversed reciprocal)')

    accum_parser = argparse.ArgumentParser(add_help=False)
    accum_parser.add_argument(
        'accum', type=str, help='[int] accumulator (final checksum)')

    default_accum_parser = argparse.ArgumentParser(add_help=False)
    default_accum_parser.add_argument(
        'accum', default='0', type=str, nargs='?',
        help='[int] starting accumulator [default: 0]')

    outfile_parser = argparse.ArgumentParser(add_help=False)
    outfile_parser.add_argument(
        '-o', '--outfile',
        metavar="f",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Output to a file instead of stdout")

    infile_parser = argparse.ArgumentParser(add_help=False)
    subparser_group = infile_parser.add_mutually_exclusive_group()
    subparser_group.add_argument(
        '-i', '--infile',
        metavar="f",
        type=argparse.FileType('rb'),
        default=sys.stdin,
        help="Input from a file instead of stdin")
    subparser_group.add_argument(
        '-s', '--str',
        metavar="s",
        type=str,
        default='',
        dest='instr',
        help="Use a string as input")

    subparsers = parser.add_subparsers(required=True, metavar='action')
    subparser = subparsers.add_parser(
        'poly', aliases=['p'],
        parents=[outfile_parser, default_poly_parser],
        help="print the polynomial, useful for converting between forms")
    subparser.set_defaults(func=poly_callback)

    subparser = subparsers.add_parser(
        'table', aliases=['t'],
        parents=[outfile_parser, default_poly_parser],
        help="generate a lookup table for a polynomial")
    subparser.set_defaults(func=table_callback)

    subparser = subparsers.add_parser(
        'reverse', aliases=['r'], parents=[
            outfile_parser,
            desired_poly_parser,
            default_accum_parser,
            default_poly_parser],
        help="find a patch that causes the CRC32 checksum to become a desired value")
    subparser.set_defaults(func=reverse_callback)

    subparser = subparsers.add_parser(
        'undo', aliases=['u'],
        parents=[
            outfile_parser,
            accum_parser,
            default_poly_parser,
            infile_parser],
        help="rewind a CRC32 checksum")
    subparser.add_argument(
        '-n', '--len', metavar='l',
        type=str,
        default='0', help='[int] number of bytes to rewind [default: 0]')
    subparser.set_defaults(func=undo_callback)

    subparser = subparsers.add_parser(
        'calc', aliases=['c'],
        parents=[
            outfile_parser,
            default_accum_parser,
            default_poly_parser,
            infile_parser],
        help="calculate the CRC32 checksum")
    subparser.set_defaults(func=calc_callback)

    return parser


def poly_callback():
    poly = get_poly()
    out('Reversed (lsbit-first)')
    out_num(poly)
    out('Normal (msbit-first)')
    out_num(reverseBits(poly))
    r = reciprocal(poly)
    out('Reversed reciprocal (Koopman notation)')
    out_num(reverseBits(r))
    out('Reciprocal')
    out_num(r)


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
        text = ''
        if all(p in permitted_characters for p in patch):
            text = '{}{}{}{} '.format(*map(chr, patch))
        out('4 bytes: {}{{0x{:02x}, 0x{:02x}, 0x{:02x}, 0x{:02x}}}'.format(text, *patch))
        checksum = calc(patch, accum)
        out('verification checksum: 0x{:08x} ({})'.format(
            checksum, 'OK' if checksum == desired else 'ERROR'))

    def print_permitted_reverse(patch):
            patches = findReverse(desired, calc(patch, accum))
            for last_4_bytes in patches:
                if all(p in permitted_characters for p in last_4_bytes):
                    patch2 = patch + last_4_bytes
                    out('{} bytes: {} ({})'.format(
                        len(patch2),
                        ''.join(map(chr, patch2)),
                        'OK' if calc(patch2, accum) == desired else 'ERROR'))

    # 5-byte alphanumeric patches
    for i in permitted_characters:
        print_permitted_reverse((i,))
    # 6-byte alphanumeric patches
    for i in permitted_characters:
        for j in permitted_characters:
            print_permitted_reverse((i, j))


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
        out_num(solution)


def calc_callback():
    # initialize tables
    init_tables(get_poly(), False)
    # calculate checksum
    accum = parse_dword(args.accum)
    data = get_input()
    out('data len: {0}'.format(len(data)))
    out('')
    out_num(calc(data, accum))


def main(argv=None):
    ''' Runs the program and handles command line options '''
    parser = get_parser()

    # Parse arguments and run the function
    global args
    args = parser.parse_args(argv)
    args.func()

if __name__ == '__main__':
    main()  # pragma: no cover
