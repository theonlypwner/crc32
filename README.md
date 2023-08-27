# CRC32 Tools

[![Build Status](https://github.com/theonlypwner/crc32/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/theonlypwner/crc32/actions/workflows/test.yml)
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
    poly (p)   print the polynomial, useful for converting between forms
    table (t)  generate a lookup table for a polynomial
    reverse (r)
               find a patch that causes the CRC32 checksum to become a desired value
    undo (u)   rewind a CRC32 checksum
    calc (c)   calculate the CRC32 checksum

options:
  -h, --help   show this help message and exit
```

## References

- Calculating Reverse CRC http://www.danielvik.com/2010/10/calculating-reverse-crc.html
- Finding Reverse CRC Patch with Readable Characters http://www.danielvik.com/2012/01/finding-reverse-crc-patch-with-readable.html
- Rewinding CRC - Calculating CRC backwards http://www.danielvik.com/2013/07/rewinding-crc-calculating-crc-backwards.html
