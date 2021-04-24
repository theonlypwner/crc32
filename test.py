#!/usr/bin/env python
# Unit Tests for Victor's CRC32 tools

import crc32
from subprocess import call
import test_data
import unittest


class ParserFunctions(unittest.TestCase):

    def setUp(self):
        crc32.init_tables(0xedb88320)

    # Tests
    def test_commandline(self):
        crc32.testing = True
        for c in test_data.commandline:
            crc32.main(c)

    def test_flip(self):
        for k in test_data.flip:
            self.assertEqual(crc32.reverseBits(k), test_data.flip[k])
            self.assertEqual(crc32.reverseBits(crc32.reverseBits(k)), k)
            self.assertEqual(
                crc32.reverseBits(crc32.reverseBits(test_data.flip[k])), test_data.flip[k])

    def test_reciprocal(self):
        for k in test_data.reciprocal:
            self.assertEqual(crc32.reciprocal(k), test_data.reciprocal[k])
            self.assertEqual(crc32.reciprocal(test_data.reciprocal[k]), k)

    def test_table(self):
        old_table = crc32.table  # save table
        for k in test_data.tables:
            crc32.init_tables(k, False)
            self.assertEqual(crc32.table, test_data.tables[k])
        crc32.table = old_table  # restore backup
        # reverse table for 0xedb88320 only
        self.assertEqual(
            list(map(lambda x: x[0], crc32.table_reverse)), test_data.edb88320_reverse)

    def test_undo_one_solution(self):
        for c in test_data.calc:
            bytes = tuple(map(ord, c[2]))
            self.assertEqual(list(crc32.rewind(c[0], bytes)), [c[1]])

    def test_undo_multiple_solutions(self):
        pass

    def test_calc(self):
        for c in test_data.calc:
            bytes = tuple(map(ord, c[2]))
            self.assertEqual(crc32.calc(bytes, c[1]), c[0])
        for c in test_data.calc4:
            self.assertEqual(crc32.calc(c[2:], c[1]), c[0])

    def test_reverse(self):
        for r in test_data.calc4:
            self.assertEqual(list(crc32.findReverse(*r[:2])), [r[2:]])

if __name__ == '__main__':
    unittest.main()
