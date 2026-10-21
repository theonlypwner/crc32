# CRC32 Tools

[![Build Status](https://github.com/theonlypwner/crc32/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/theonlypwner/crc32/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/theonlypwner/crc32/badge.svg)](https://coveralls.io/github/theonlypwner/crc32)
[![PyPI](https://img.shields.io/pypi/v/crc32.svg)](https://pypi.org/project/crc32/)

## License

This project is licensed under the GPL v3 license.

## Usage

Run the command line to see usage instructions:

```console
$ crc32.py -h
usage: crc32.py [-h] action ...

Reverse, undo, and calculate CRC32 checksums

positional arguments:
  action
    poly (p)     print the polynomial, useful for converting between forms
    table (t)    generate a lookup table for a polynomial
    reverse (r)  find a patch that causes the CRC32 checksum to become a desired value
    undo (u)     rewind a CRC32 checksum
    calc (c)     calculate the CRC32 checksum
    combine      combine CRC32 checksums

options:
  -h, --help     show this help message and exit

$ crc32.py poly -h
usage: crc32.py poly [-h] [-o f] [-m | -l] [-r] [poly]

positional arguments:
  poly                  [int] polynomial [default: 0xEDB88320]

options:
  -h, --help            show this help message and exit
  -o, --outfile f       Output to a file instead of stdout
  -m, --msbit, --normal
                        treat the polynomial as normal (msbit-first)
  -l, --lsbit, --reversed
                        treat the polynomial as reversed (lsbit-first)
                        [default]
  -r, --reciprocal      treat the polynomial as reciprocal (Koopman notation
                        is reversed reciprocal)

$ crc32.py table -h
usage: crc32.py table [-h] [-o f] [-m | -l] [-r] [poly]

positional arguments:
  poly                  [int] polynomial [default: 0xEDB88320]

options:
  -h, --help            show this help message and exit
  -o, --outfile f       Output to a file instead of stdout
  -m, --msbit, --normal
                        treat the polynomial as normal (msbit-first)
  -l, --lsbit, --reversed
                        treat the polynomial as reversed (lsbit-first)
                        [default]
  -r, --reciprocal      treat the polynomial as reciprocal (Koopman notation
                        is reversed reciprocal)

$ crc32.py reverse -h
usage: crc32.py reverse [-h] [-o f] [-m | -l] [-r] desired [accum] [poly]

positional arguments:
  desired               [int] desired checksum
  accum                 [int] starting accumulator [default: 0]
  poly                  [int] polynomial [default: 0xEDB88320]

options:
  -h, --help            show this help message and exit
  -o, --outfile f       Output to a file instead of stdout
  -m, --msbit, --normal
                        treat the polynomial as normal (msbit-first)
  -l, --lsbit, --reversed
                        treat the polynomial as reversed (lsbit-first)
                        [default]
  -r, --reciprocal      treat the polynomial as reciprocal (Koopman notation
                        is reversed reciprocal)

$ crc32.py undo -h
usage: crc32.py undo [-h] [-o f] [-m | -l] [-r] [-O OFFSET] [-i f | -s s]
                     [-n l]
                     accum [poly]

positional arguments:
  accum                 [int] accumulator (final checksum)
  poly                  [int] polynomial [default: 0xEDB88320]

options:
  -h, --help            show this help message and exit
  -o, --outfile f       Output to a file instead of stdout
  -m, --msbit, --normal
                        treat the polynomial as normal (msbit-first)
  -l, --lsbit, --reversed
                        treat the polynomial as reversed (lsbit-first)
                        [default]
  -r, --reciprocal      treat the polynomial as reciprocal (Koopman notation
                        is reversed reciprocal)
  -O, --offset OFFSET   Offset (in bytes) to start reading from input file
                        [default: 0]
  -i, --infile f        Input from a file instead of stdin
  -s, --str s           Use a string as input
  -n, --len l           [int] number of bytes to rewind [default: 0]

$ crc32.py calc -h
usage: crc32.py calc [-h] [-o f] [-m | -l] [-r] [-O OFFSET] [-i f | -s s]
                     [accum] [poly]

positional arguments:
  accum                 [int] starting accumulator [default: 0]
  poly                  [int] polynomial [default: 0xEDB88320]

options:
  -h, --help            show this help message and exit
  -o, --outfile f       Output to a file instead of stdout
  -m, --msbit, --normal
                        treat the polynomial as normal (msbit-first)
  -l, --lsbit, --reversed
                        treat the polynomial as reversed (lsbit-first)
                        [default]
  -r, --reciprocal      treat the polynomial as reciprocal (Koopman notation
                        is reversed reciprocal)
  -O, --offset OFFSET   Offset (in bytes) to start reading from input file
                        [default: 0]
  -i, --infile f        Input from a file instead of stdin
  -s, --str s           Use a string as input

$ crc32.py combine -h
usage: crc32.py combine [-h] [-o f] [-m | -l] [-r]
                        accum checksum len [n] [poly]

positional arguments:
  accum                 [int] accumulator (initial checksum)
  checksum              [int] checksum of message
  len                   [int] length of message
  n                     [int] number of times to append message [default: 1]
  poly                  [int] polynomial [default: 0xEDB88320]

options:
  -h, --help            show this help message and exit
  -o, --outfile f       Output to a file instead of stdout
  -m, --msbit, --normal
                        treat the polynomial as normal (msbit-first)
  -l, --lsbit, --reversed
                        treat the polynomial as reversed (lsbit-first)
                        [default]
  -r, --reciprocal      treat the polynomial as reciprocal (Koopman notation
                        is reversed reciprocal)
```

## References

- Calculating Reverse CRC http://www.danielvik.com/2010/10/calculating-reverse-crc.html
- Finding Reverse CRC Patch with Readable Characters http://www.danielvik.com/2012/01/finding-reverse-crc-patch-with-readable.html
- Rewinding CRC - Calculating CRC backwards http://www.danielvik.com/2013/07/rewinding-crc-calculating-crc-backwards.html
