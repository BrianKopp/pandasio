import unittest
from numpy import float64
from pandasio.pandabar import _PandaBar, _get_panda_bar_info_dtype
from pandasio.exceptions import IdentifierByteRepresentationError
from pandasio.utils.exceptions import NumBytesForStringInvalidError
from pandasio.utils.numpy_utils import NumpyTypeChars


class TestPandaBar(unittest.TestCase):
    def test_panda_bar_info_dtype(self):
        d = _get_panda_bar_info_dtype(4)
        self.assertEqual('identifier', d.descr[0][0])
        self.assertEqual('<U1', d.descr[0][1])

        self.assertEqual('options', d.descr[1][0])
        self.assertEqual('<u2', d.descr[1][1])

        self.assertEqual('bytes_per_point', d.descr[2][0])
        self.assertEqual('|u1', d.descr[2][1])

        self.assertEqual('type_char', d.descr[3][0])
        self.assertEqual('|u1', d.descr[3][1])

        self.assertEqual('bytes_extra_information', d.descr[4][0])
        self.assertEqual('<u4', d.descr[4][1])

        for i in range(0, 32):
            self.assertEqual('def_byte_{}'.format(i + 1), d.descr[5 + i][0])
            self.assertEqual('|u1', d.descr[5 + i][1])

        d = _get_panda_bar_info_dtype(16)
        self.assertEqual('<U4', d.descr[0][1])

        d = _get_panda_bar_info_dtype(32)
        self.assertEqual('<U8', d.descr[0][1])

        # test errors
        with self.assertRaises(NumBytesForStringInvalidError):
            _get_panda_bar_info_dtype(2)
        with self.assertRaises(IdentifierByteRepresentationError):
            _get_panda_bar_info_dtype(0)
        with self.assertRaises(IdentifierByteRepresentationError):
            _get_panda_bar_info_dtype(-1)
        with self.assertRaises(NumBytesForStringInvalidError):
            _get_panda_bar_info_dtype(0.5)
        return

    def test_panda_bar_init_min_params(self):
        p = _PandaBar('data', 8, NumpyTypeChars.FLOAT)

        self.assertEqual('data', p._identifier)
        self.assertEqual(8, p._bytes_per_value)
        self.assertEqual('f', p._type_char)
        self.assertEqual(float64, p._dtype)
        self.assertEqual(False, p._is_index)

        self.assertEqual(None, p._data)
        self.assertEqual(None, p._encoded_data)

        self.assertEqual(None, p._num_points)

        self.assertEqual(True, p._use_compression)
        self.assertEqual(False, p._use_hash_table)
        self.assertEqual(False, p._use_floating_point_rounding)

        self.assertEqual(None, p._compression_dtype)
        self.assertEqual(None, p._compression_mode)
        self.assertEqual(None, p._compression_reference_value)
        self.assertEqual(None, p._compression_reference_value_dtype)

        self.assertEqual(None, p._floating_point_rounding_num_decimals)
        self.assertEqual(0, p._num_bytes_extra_information)
        return

    def test_panda_bar_init_optional_params(self):
        p = _PandaBar('data', 8, NumpyTypeChars.FLOAT, is_index=True)
        self.assertEqual(True, p._is_index)
        self.assertEqual(True, p.is_index())

        p = _PandaBar('data', 8, NumpyTypeChars.FLOAT, num_extra_bytes_required=4)
        self.assertEqual(4, p._num_bytes_extra_information)
        self.assertEqual(4, p.num_extra_bytes_required())
        return

    def test_panda_bar_encode_options(self):
        p = _PandaBar('data', 8, NumpyTypeChars.FLOAT)

        p._use_floating_point_rounding = False
        p._use_hash_table = False
        p._use_compression = False
        p._is_index = False
        self.assertEqual(0, p._encode_options())

        p._use_floating_point_rounding = False
        p._use_hash_table = False
        p._use_compression = False
        p._is_index = True
        self.assertEqual(1, p._encode_options())

        p._use_floating_point_rounding = False
        p._use_hash_table = False
        p._use_compression = True
        p._is_index = False
        self.assertEqual(2, p._encode_options())

        p._use_floating_point_rounding = False
        p._use_hash_table = False
        p._use_compression = True
        p._is_index = True
        self.assertEqual(3, p._encode_options())

        p._use_floating_point_rounding = False
        p._use_hash_table = True
        p._use_compression = False
        p._is_index = False
        self.assertEqual(4, p._encode_options())

        p._use_floating_point_rounding = False
        p._use_hash_table = True
        p._use_compression = False
        p._is_index = True
        self.assertEqual(5, p._encode_options())

        p._use_floating_point_rounding = False
        p._use_hash_table = True
        p._use_compression = True
        p._is_index = False
        self.assertEqual(6, p._encode_options())

        p._use_floating_point_rounding = False
        p._use_hash_table = True
        p._use_compression = True
        p._is_index = True
        self.assertEqual(7, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = False
        p._use_compression = False
        p._is_index = False
        self.assertEqual(8, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = False
        p._use_compression = False
        p._is_index = True
        self.assertEqual(9, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = False
        p._use_compression = True
        p._is_index = False
        self.assertEqual(10, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = False
        p._use_compression = True
        p._is_index = True
        self.assertEqual(11, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = True
        p._use_compression = False
        p._is_index = False
        self.assertEqual(12, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = True
        p._use_compression = False
        p._is_index = True
        self.assertEqual(13, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = True
        p._use_compression = True
        p._is_index = False
        self.assertEqual(14, p._encode_options())

        p._use_floating_point_rounding = True
        p._use_hash_table = True
        p._use_compression = True
        p._is_index = True
        self.assertEqual(15, p._encode_options())
        return

    def test_panda_bar_decode_options(self):
        p = _PandaBar('data', 8, NumpyTypeChars.FLOAT)

        p._decode_options(0)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(1)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertTrue(p._is_index)

        p._decode_options(2)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(3)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertTrue(p._is_index)

        p._decode_options(4)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(5)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertTrue(p._is_index)

        p._decode_options(6)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(7)
        self.assertFalse(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertTrue(p._is_index)

        p._decode_options(8)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(9)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertTrue(p._is_index)

        p._decode_options(10)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(11)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertFalse(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertTrue(p._is_index)

        p._decode_options(12)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(13)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertFalse(p._use_compression)
        self.assertTrue(p._is_index)

        p._decode_options(14)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertFalse(p._is_index)

        p._decode_options(15)
        self.assertTrue(p._use_floating_point_rounding)
        self.assertTrue(p._use_hash_table)
        self.assertTrue(p._use_compression)
        self.assertTrue(p._is_index)
        return

if __name__ == '__main__':
    unittest.main()
