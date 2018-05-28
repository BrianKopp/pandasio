import unittest
from pandasio.pandabar import _PandaBar
from pandasio.utils.numpy_utils import NumpyTypeChars


def get_blank_bytes_list():
    return [b'\x00' for _ in range(0, 32)]


def get_blank_bytes():
    return b''.join(get_blank_bytes_list())


class TestPandaBarDetailsBytes(unittest.TestCase):
    def test_panda_bar_encode_bytes(self):
        p = _PandaBar('data', 8, NumpyTypeChars.FLOAT)
        p._use_compression = False
        self.assertEqual(get_blank_bytes(), p._encode_details_bytes())

        p._use_floating_point_rounding = True
        p._floating_point_rounding_num_decimals = 4
        ret_bytes = p._encode_details_bytes()
        self.assertEqual(4, ret_bytes[0])
        return

if __name__ == '__main__':
    unittest.main()
