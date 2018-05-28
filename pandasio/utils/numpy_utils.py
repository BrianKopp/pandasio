import numpy as np
from pandasio.utils.validation import ensure_int
from pandasio.utils.exceptions import *
from typing import Union
from enum import Enum


class NumpyTypeChars(Enum):
    UNSIGNED = 'u'
    INTEGER = 'i'
    FLOAT = 'f'
    STRING = 'U'


def get_type_char_int(type_char: Union[NumpyTypeChars, str, int]) -> int:
    """
    Gets the integer ord() of the character
    :param type_char: NumpyTypeChars enum, string, or integer to convert into an integer.
    :return: integer corresponding to the char (or int)
    """
    if isinstance(type_char, NumpyTypeChars):
        return ord(type_char.value)
    if isinstance(type_char, str):
        return ord(type_char)
    if isinstance(type_char, int):
        return type_char
    try:
        return ensure_int(type_char)
    except:
        raise CharConversionException(
            'Could not convert type: {} for value: {}'.format(
                type(type_char),
                type_char
            )
        )


def get_type_char_char(type_char: Union[NumpyTypeChars, str, int]) -> str:
    """
    Gets the character representation of the char or int.
    :param type_char: NumpyTypeChars enum, string, or integer to convert into an integer.
    :return: string corresponding to parameter
    """
    if isinstance(type_char, NumpyTypeChars):
        return type_char.value
    if isinstance(type_char, int):
        return chr(type_char)
    if isinstance(type_char, float):
        return chr(ensure_int(type_char))
    if isinstance(type_char, str):
        if len(type_char) == 1:
            return type_char
        else:
            raise CharConversionException('Could not convert string \'{}\'into '
                                          'char because its length was > 1'.format(type_char))
    # last ditch, try to cast it as int
    try:
        return chr(int(type_char))
    except TypeError:
        raise CharConversionException(
            'Could not convert type: {} for value: {}'.format(
                type(type_char),
                type_char
            )
        )


def get_numpy_type(type_char: Union[NumpyTypeChars, str, int], size: int) -> np.dtype:
    """
    Gets the numpy data type corresponding to the type char ('i', 'u', 'f')
    and size in bits of the value.
    :param type_char: NumpyTypeChars, string, or integer corresponding to char in ('i', 'u', 'f')
    :param size: number of BITS in the data
    :return: numpy dtype
    """
    type_char = get_type_char_char(type_char)
    size = ensure_int(size)
    if size <= 0:
        raise DataSizeNotPositiveError('Cannot make a negative-sized numpy type')
    if type_char == NumpyTypeChars.INTEGER.value:
        if size == 8:
            return np.int8
        elif size == 16:
            return np.int16
        elif size == 32:
            return np.int32
        elif size == 64:
            return np.int64
    elif type_char == NumpyTypeChars.UNSIGNED.value:
        if size == 8:
            return np.uint8
        elif size == 16:
            return np.uint16
        elif size == 32:
            return np.uint32
        elif size == 64:
            return np.uint64
    elif type_char == NumpyTypeChars.FLOAT.value:
        if size == 16:
            return np.float16
        elif size == 32:
            return np.float32
        elif size == 64:
            return np.float64
    elif type_char == NumpyTypeChars.STRING.value:
        try:
            num_chars = ensure_int(size / 32)
        except NotIntegerException:
            raise NumBytesForStringInvalidError('Could not get numpy type for string.'
                                                ' Number of bits must be multiple of 32')
        return '<U{}'.format(num_chars)
    raise ValueError(
        'Could not find match for char {} and size {}'.format(
            type_char,
            size
        )
    )
