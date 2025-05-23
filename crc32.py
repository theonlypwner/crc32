#!/usr/bin/env python
# Command line script for CRC32 tools

import argparse
import sys

from crc32 import CRC32, CRC32Reverse, combine, reverse_bits, reciprocal

def get_poly(args) -> int:
    poly = parse_dword(args.poly)
    if args.msb:
        poly = reverse_bits(poly)
    if args.reciprocal:
        poly = reverse_bits(reciprocal(poly))

    if poly & 0x80000000 == 0:
        suggested = poly | 0x80000000
        print('WARNING: polynomial degree ({0}) != 32'.format(poly.bit_length()), file=args.outfile)
        print('         instead, try', file=args.outfile)
        print('         0x{0:08x} (reversed/lsbit-first)'.format(suggested), file=args.outfile)
        print('         0x{0:08x} (normal/msbit-first)'.format(reverse_bits(suggested)), file=args.outfile)

    return poly

def get_input(args):
    if args.instr:
        return args.instr.encode('utf-8')
    with args.infile as f:
        if hasattr(args, 'offset') and args.offset > 0:
            f.seek(args.offset)
        return f.read()

def parse_dword(x: str) -> int:
    return int(x, 0) & 0xFFFFFFFF


def print_num(num, **kwargs):
    ''' Write a numeric result in various forms '''
    print('hex: 0x{0:08x}'.format(num), **kwargs)
    print('dec:   {0:d}'.format(num), **kwargs)
    print('oct: 0o{0:011o}'.format(num), **kwargs)
    print('bin: 0b{0:032b}'.format(num), **kwargs)

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

    combine_parser = argparse.ArgumentParser(add_help=False)
    combine_parser.add_argument(
        'accum', type=str, help='[int] accumulator (initial checksum)')
    combine_parser.add_argument(
        'checksum', type=str,
        help='[int] checksum of message')
    combine_parser.add_argument(
        'len', type=str,
        help='[int] length of message')
    combine_parser.add_argument(
        'n', default='1', type=str, nargs='?',
        help='[int] number of times to append message [default: 1]')

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
    infile_parser.add_argument(
        '--offset', '-O',
        type=int,
        default=0,
        help='Offset (in bytes) to start reading from input file [default: 0]')

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

    subparser = subparsers.add_parser(
        'combine',
        parents=[
            outfile_parser,
            combine_parser,
            default_poly_parser],
        help="combine CRC32 checksums")
    subparser.set_defaults(func=combine_callback)

    return parser


def poly_callback(args):
    poly = get_poly(args)
    print('Reversed (lsbit-first)', file=args.outfile)
    print_num(poly, file=args.outfile)
    print('Normal (msbit-first)', file=args.outfile)
    print_num(reverse_bits(poly), file=args.outfile)
    r = reciprocal(poly)
    print('Reversed reciprocal (Koopman notation)', file=args.outfile)
    print_num(reverse_bits(r), file=args.outfile)
    print('Reciprocal', file=args.outfile)
    print_num(r, file=args.outfile)


def table_callback(args):
    crc32 = CRC32(get_poly(args))
    # print table
    print('[{0}]'.format(', '.join(map('0x{0:08x}'.format, crc32.table))), file=args.outfile)


def reverse_callback(args):
    permitted_characters = set(
        b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890_')  # \w

    crc32_reverse = CRC32Reverse(get_poly(args))
    # find reverse bytes
    desired = parse_dword(args.desired)
    accum = parse_dword(args.accum)
    # 4-byte patch
    patches = crc32_reverse.find_reverse(desired, accum)
    for patch in patches:
        text = ''
        if all(p in permitted_characters for p in patch):
            text = patch.decode() + ' '
        print('4 bytes: {}{{0x{:02x}, 0x{:02x}, 0x{:02x}, 0x{:02x}}}'.format(text, *patch), file=args.outfile)
        checksum = crc32_reverse.calc(patch, accum)
        print('verification checksum: 0x{:08x} ({})'.format(
            checksum, 'OK' if checksum == desired else 'ERROR'), file=args.outfile)

    def print_permitted_reverse(patch: bytes):
        patches = crc32_reverse.find_reverse(desired, crc32_reverse.calc(patch, accum))
        for last_4_bytes in patches:
            if all(p in permitted_characters for p in last_4_bytes):
                patch2 = patch + last_4_bytes
                print('{} bytes: {} ({})'.format(
                    len(patch2),
                    patch2.decode(),
                    'OK' if crc32_reverse.calc(patch2, accum) == desired else 'ERROR'), file=args.outfile)

    # 5-byte alphanumeric patches
    for i in permitted_characters:
        print_permitted_reverse(bytes([i]))
    # 6-byte alphanumeric patches
    for i in permitted_characters:
        for j in permitted_characters:
            print_permitted_reverse(bytes([i, j]))


def undo_callback(args):
    crc32_reverse = CRC32Reverse(get_poly(args))
    # calculate checksum
    accum = parse_dword(args.accum)
    maxlen = int(args.len, 0)
    data = get_input(args)
    if not 0 < maxlen <= len(data):
        maxlen = len(data)
    print('rewinded {0}/{1} ({2:.2f}%)'.format(maxlen, len(data),
        maxlen * 100.0 / len(data) if len(data) else 100), file=args.outfile)
    for solution in crc32_reverse.rewind(data[-maxlen:], accum):
        print('', file=args.outfile)
        print_num(solution, file=args.outfile)


def calc_callback(args):
    crc32 = CRC32(get_poly(args))
    # calculate checksum
    accum = parse_dword(args.accum)
    data = get_input(args)
    print('data len: {0}'.format(len(data)), file=args.outfile)
    print('', file=args.outfile)
    print_num(crc32.calc(data, accum), file=args.outfile)


def combine_callback(args):
    c1 = parse_dword(args.accum)
    c2 = parse_dword(args.checksum)
    l2 = parse_dword(args.len)
    n = int(args.n, 0)

    print_num(combine(c1, c2, l2, n, get_poly(args)), file=args.outfile)


def main(argv=None):
    ''' Runs the program and handles command line options '''
    parser = get_parser()

    # Parse arguments and run the function
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == '__main__':
    main()
