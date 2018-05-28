from numpy import array, uint16, dtype, uint8, uint32, fromfile, frombuffer
from collections import namedtuple
from typing import Union
from pandasio.utils.numpy_utils import get_numpy_type, get_type_char_char,\
    get_type_char_int, NumpyTypeChars
from pandasio.utils.numpy_compression import round_array_returning_integers, compress_array, decompress_array
from pandasio.utils.exceptions import DataSizeNotPositiveError, NumBytesForStringInvalidError
from pandasio.exceptions import IdentifierByteRepresentationError


ByteResultTuple = namedtuple('ByteResultTuple', ['num_bytes', 'byte_code'])
_COMPRESSION_MODE_ELEMENT_WISE = 'e'
NUM_BYTES_PER_DEFINITION_WITHOUT_IDENTIFIER = 40


def _get_panda_bar_info_dtype(num_bytes_for_identifier: int) -> dtype:
    """
    gets the dtype for the PandaBar class's info bytes
    :param num_bytes_for_identifier: number of bytes needed to store identifier
    # :param identifier_is_string: whether the identifier is a string, else int
    extra information for PandaBar
    :return: numpy dtype
    """
    dtype_list = []
    # if identifier_is_string:
    try:
        dtype_list.append((
            'identifier',
            get_numpy_type(NumpyTypeChars.STRING, 8 * num_bytes_for_identifier)
        ))
    except DataSizeNotPositiveError:
        raise IdentifierByteRepresentationError('Cannot have negative bytes per identifier')
    except NumBytesForStringInvalidError:
        raise NumBytesForStringInvalidError('Number of bytes for identifier must be multiple of 4 bytes')

    #else:
    #dtype_list.append((
    #    'identifier',
    #    get_numpy_type(NumpyTypeChars.UNSIGNED, 8 * num_bytes_for_identifier)
    # ))
    # 2 bytes of options
    dtype_list.append(('options', uint16))
    # bytes per point
    dtype_list.append(('bytes_per_point', uint8))
    # type char
    dtype_list.append(('type_char', uint8))
    # 4-bytes indicating length of extra information
    dtype_list.append(('bytes_extra_information', uint32))
    # 32-bytes of details
    dtype_list.extend([('def_byte_{}'.format(i + 1), uint8) for i in range(0, 32)])
    return dtype(dtype_list)


class _PandaBar:
    """
    Class serving binary i/o of a pandas DataFrame's column (Series).
    Intended to be accessed only internally by the PandaCage class.
    """
    def __init__(self, identifier: str, bytes_per_value: int, type_char: Union[NumpyTypeChars, int, str],
                 is_index: bool = False, options: int = None,
                 num_extra_bytes_required: int = None, details_bytes: bytes = None, data: array = None):
        """
        Initializes a PandaBar class object
        :param identifier: unique id (string) for data
        :param bytes_per_value: int, size of item of data in bytes (uncompressed)
        :param type_char: NumpyTypeChar or int/string describing type of data
        :param is_index: boolean indicating whether this PandaBar is an index column
        :param options: 16-bit integer containing options flags in the bits
        :param num_extra_bytes_required: integer describing number of extra bytes required outside details
        :param details_bytes: 32-bytes of detail bytes for the PandaBar
        :param data: numpy array containing uncompressed data.
        """
        self._identifier = identifier
        self._bytes_per_value = bytes_per_value
        self._type_char = get_type_char_char(type_char)
        self._dtype = get_numpy_type(self._type_char, 8*self._bytes_per_value)
        self._is_index = is_index

        self._data = None  # numpy array
        self._encoded_data = None  # numpy array

        # other metrics used as helpers
        self._num_points = None

        # options
        self._use_compression = True
        self._use_hash_table = False
        self._use_floating_point_rounding = False

        # compression options
        self._compression_dtype = None
        self._compression_mode = None
        self._compression_reference_value = None
        self._compression_reference_value_dtype = None

        # floating point rounding
        self._floating_point_rounding_num_decimals = None

        # additional information that may be needed to detail options
        self._num_bytes_extra_information = 0 if num_extra_bytes_required is None else num_extra_bytes_required

        if options is not None:
            self._decode_options(options)
        if details_bytes is not None:
            self._decode_details_bytes(details_bytes)
        if data is not None:
            self.set_data(data)
        return

    @classmethod
    def decode_panda_bars_definitions_from_bytes(cls, from_bytes: bytes, num_bytes_for_identifier: int) -> dict:
        """
        Reads bytes and returns a list of PandaBar objects
        :param from_bytes: binary bytes to read from
        :param num_bytes_for_identifier: integer specifying number of bytes per identifier
        # :param identifiers_are_strings: whether or not the ids are strings
        :return: dictionary like {identifier : PandaBar}
        """
        raw_bars = frombuffer(
            from_bytes,
            dtype=_get_panda_bar_info_dtype(
                num_bytes_for_identifier
            )
        )
        bars = [
            _PandaBar(
                b['identifier'],
                b['bytes_per_point'],
                b['type_char'],
                options=b['options'],
                num_extra_bytes_required=b['bytes_extra_information'],
                details_bytes=b''.join([b['def_byte_{}'.format(i + 1)] for i in range(0, 32)])
            )
            for b in raw_bars
        ]
        return dict([(b._identifier, b) for b in bars])

    def encode_info(self, num_bytes_for_identifier: int) -> ByteResultTuple:
        """
        encodes the PandaBar information into binary form
        :param num_bytes_for_identifier: number of bytes required to store identifier
        # :param identifier_is_string: boolean indicating whether to cast identifier to string
        :return: named tuple of ByteResultTuple
        """
        option_integer = self._encode_options()
        encode_id = self._identifier
        info = array(
            [(
                encode_id,
                option_integer,
                self._bytes_per_value,
                get_type_char_int(self._type_char),
                self._num_bytes_extra_information
            )],
            dtype=_get_panda_bar_info_dtype(
                num_bytes_for_identifier
            )
        )

        self._encode_data()

        details_bytes = self._encode_details_bytes()
        ret_bytes = info.tobytes() + details_bytes
        return ByteResultTuple(num_bytes=len(ret_bytes), byte_code=ret_bytes)

    def data_to_file(self, file_handle) -> int:
        """
        Sends the binary encoded data to the file handle at the correct seek position
        :param file_handle: file handle in mode 'wb' at the correct seek position
        :return: int number of bytes written
        """
        self._encode_data()
        self._encoded_data.tofile(file_handle)
        return self._encoded_data.nbytes

    def data_from_file(self, file_handle, num_points: int) -> int:
        """
        reads the encoded data from the file handle
        :param file_handle: handle in 'rb' mode at correct seek position
        :param num_points: number of points that are in the PandaCage storage
        :return: integer, number of bytes read from the file
        """
        self._num_points = num_points
        read_num_points = num_points
        read_dtype = self._dtype
        if self._use_compression:
            read_dtype = self._compression_dtype
            if _COMPRESSION_MODE_ELEMENT_WISE in self._compression_mode:
                read_num_points -= 1
        self._encoded_data = fromfile(
            file_handle,
            read_dtype,
            count=read_num_points
        )
        return self._encoded_data.nbytes

    def set_data(self, data: array):
        """
        Sets the internal data array of the PandaBar.
        Casts the array into type of prev
        :param data: numpy array holding the data
        :return: None, populates class internals
        """
        self._data = data.astype(self._dtype)
        self._num_points = self._data.size
        self._encoded_data = None
        return

    def get_data(self) -> array:
        """
        Gets the data from the PandaBar
        :return: numpy array containing data
        """
        if self._data is None:
            self._decode_data()
        return array(self._data)  # makes a copy

    def prepare_for_write(self):
        """
        Method to perform any tasks needed to prepare for writing. This entails doing any
        compression or other algorithms on the data
        :return: None, raises exception if there are any issues
        """
        raise NotImplementedError()  # TODO

    def validate(self) -> bool:
        """
        runs validation logic on data
        :return: True, or raises exception
        """
        raise NotImplementedError()  # TODO

    def is_index(self) -> bool:
        """
        Returns whether or not the bar is an index column
        :return: boolean
        """
        return self._is_index

    def num_points(self) -> int:
        """
        gets the number of points in the data
        :return: number of points
        """
        return self._num_points

    def num_extra_bytes_required(self) -> int:
        """
        :return: returns the number of extra bytes required for compression tables etc.
        """
        return self._num_bytes_extra_information

    def _encode_options(self) -> uint16:
        """
        Encodes 16 bit options onto an unsigned 16-bit numpy integer
        :return: numpy 16-bit integer
        """
        # start with the left-most bits and work right
        options = 0
        options |= 1 if self._use_floating_point_rounding else 0
        options <<= 1
        options |= 1 if self._use_hash_table else 0
        options <<= 1
        options |= 1 if self._use_compression else 0
        options <<= 1
        options |= 1 if self._is_index else 0
        return uint16(options)

    def _decode_options(self, from_int: int):
        """
        Decodes a 16-bit integer containing options
        :param from_int: 16-bit unsigned integer to decode from
        :return: None, populates class internals
        """
        # starting with the right-most bits and working left
        self._is_index = True if (from_int >> 0) & 1 else False
        self._use_compression = True if (from_int >> 1) & 1 else False
        self._use_hash_table = True if (from_int >> 2) & 1 else False
        self._use_floating_point_rounding = True if (from_int >> 3) & 1 else False
        return

    def _encode_details_bytes(self) -> bytes:
        """
        Encodes 32-bytes of values to send to binary
        :return: bytes, 32 long, numpy uint8 type
        """
        ret_bytes = [b'\x00' for _ in range(0, 32)]
        counter = 0
        if self._use_compression:
            ret_bytes[counter] = get_type_char_int(self._compression_mode).to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = self._compression_dtype.itemsize.to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = get_type_char_int(self._compression_dtype.kind).to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = self._compression_reference_value_dtype.itemsize.to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = get_type_char_int(self._compression_reference_value_dtype.kind).to_bytes(1, 'little')
            counter += 1
            reference_value_bytes = array(
                [self._compression_reference_value],
                dtype=self._compression_reference_value_dtype
            ).tobytes()
            for i in range(0, len(reference_value_bytes)):
                ret_bytes[counter] = reference_value_bytes[i].to_bytes(1, 'little')
                counter += 1
        if self._use_floating_point_rounding:
            ret_bytes[counter] = self._floating_point_rounding_num_decimals.to_bytes(1, 'little')
            counter += 1
        return b''.join(ret_bytes)

    def _decode_details_bytes(self, from_bytes: bytes):
        """
        Decodes the 32-bytes of binary data to populate class internals
        :param from_bytes: 32-bytes of data to decode
        :return: None, populates class internals
        """
        counter = 0
        if self._use_compression:
            compression_info = frombuffer(from_bytes[counter:(counter+5)], dtype=uint8, count=5)
            counter += 5
            self._compression_mode = get_type_char_char(compression_info[0])
            bytes_per_value = compression_info[1]
            type_char = get_type_char_char(compression_info[2])
            self._compression_dtype = get_numpy_type(type_char, bytes_per_value)
            self._compression_reference_value_dtype = get_numpy_type(
                get_type_char_char(compression_info[4]),
                compression_info[3] * 8
            )
            ref_value_bytes = self._bytes_per_value
            self._compression_reference_value = frombuffer(
                from_bytes[counter:counter+ref_value_bytes],
                dtype=self._compression_reference_value_dtype,
                count=1
            )[0]
            counter += ref_value_bytes
        if self._use_floating_point_rounding:
            self._floating_point_rounding_num_decimals = from_bytes[counter]
            counter += 1
        return

    def _encode_data(self):
        """
        Performs compression and alteration on data to produce data-set
        that will be written to binary form
        :return: None
        """
        if self._encoded_data is not None:
            return
        self._encoded_data = self._data

        # TODO datetimes should be handled here

        if self._use_floating_point_rounding:
            self._encoded_data = round_array_returning_integers(
                self._encoded_data,
                self._floating_point_rounding_num_decimals
            )
        if self._use_compression:
            self._compression_reference_value_dtype = self._encoded_data.dtype
            mode = 'm' if self._compression_mode is None else self._compression_mode
            compression_result = compress_array(self._encoded_data, mode)
            self._compression_mode = mode
            self._compression_dtype = compression_result.numpy_array.dtype
            self._encoded_data = compression_result.numpy_array
            self._compression_reference_value = compression_result.reference_value
        return

    def _decode_data(self):
        """
        Decodes data from internal encoded data
        :return: None, populates class internals
        """
        self._data = self._encoded_data
        if self._use_compression:
            self._data = decompress_array(
                self._data,
                self._compression_mode,
                self._compression_reference_value
            ).astype(self._dtype)
        if self._use_floating_point_rounding:
            self._data /= pow(10, self._floating_point_rounding_num_decimals)
        self._num_points = self._data.size
        return
