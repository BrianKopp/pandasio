from numpy import array, isnan, frombuffer, bitwise_and, full, count_nonzero,\
    frexp, amin, amax, ediff1d, cumsum, insert, add, around
from numpy import dtype, float16, float32, uint8, int8, int64
from collections import namedtuple
from pandasio.utils.binary import determine_required_bytes_unsigned_integer, determine_required_bytes_signed_integer
from pandasio.utils.exceptions import ArrayNotFloatException, CompressionModeInvalidError, CompressionError, \
    NotIntegerException
from pandasio.utils.numpy_utils import get_numpy_type
from pandasio.utils.validation import ensure_int

CompressionResult = namedtuple('CompressionResult', ['numpy_array', 'reference_value'])


def compress_float_array(arr: array) -> array:
    """
    Takes a numpy array with dtype.kind = 'f',
    and checks to see which bits are used in the float,
    then returns the compressed array as a float16, float32, or float64
    in a loss-less format
    :param arr: numpy array of floats
    :return: compressed array
    """
    if arr.dtype.kind != 'f':
        raise ArrayNotFloatException
    # filter out NaN values
    not_nan = arr[~isnan(arr)]
    if not_nan.size == 0:
        return arr.astype(float16)

    # put the array into a byte array
    bytes_dtype = dtype([('byte_{}'.format(i), uint8) for i in range(0, arr.itemsize)])
    byte_array = frombuffer(not_nan.tobytes(), dtype=bytes_dtype)
    byte_arr_len = byte_array['byte_0'].shape[0]
    # mantissa is X bits long, with left-right having value of
    # 0.5 * 2^(i)
    big_endian_nth_byte = dict(
        [('byte_{}'.format(i), 'byte_{}'.format(arr.itemsize - i - 1)) for i in range(0, arr.itemsize)]
    )
    if arr.itemsize == 8:
        # numpy 64-bit float has 52 bits of mantissa
        # in order to compress to 32-bit float, we need to pack mantissa into
        # 23 bits. This means we need 52-23=29 bits of zeros on the end.
        # that means that the 3 right-most bytes must be zero
        # and that the 4th right-most byte must have 5 right-most bits = 0
        # we'll check this by seeing if '0001 1111' & byte == 0. (0001 1111 is 31)
        # finally, we'll need to check the exponent compression

        # first check the 4th right-most byte. it's most likely to have the
        # significant bits we can't drop
        bitwise_reduction = bitwise_and(
            byte_array[big_endian_nth_byte['byte_4']],
            full(byte_arr_len, 31, dtype=int8)
        )
        if count_nonzero(bitwise_reduction) > 0:
            return arr
        # next check the remaining bits
        if count_nonzero(byte_array[big_endian_nth_byte['byte_5']]) > 0 \
            or count_nonzero(byte_array[big_endian_nth_byte['byte_6']]) > 0 \
                or count_nonzero(byte_array[big_endian_nth_byte['byte_7']]) > 0:
            return arr
        # finally check the exponent. in 64-bit float, exponent can be +/- 1024.
        # in 32-bit float, exponent can be +/-128
        # do this the easy way and get this from numpy
        bitwise_exponent = frexp(not_nan)[1]
        if amax(bitwise_exponent) > 128 or amin(bitwise_exponent) < -128:
            return arr

        # else, we're good to go onto 32-bit reduction
        return compress_float_array(arr.astype(float32))

    if arr.itemsize == 4:
        # numpy 32-bit float has 23 bits of mantissa.
        # numpy 16-bit float has 10 bits mantissa.
        # we require the right-most 13-bits to be zero. byte_3 (4th byte)
        # must be all zero, and byte_2 (3rd byte) must have
        # the 5 right-most bits be zero. We will bitwise-and against 31
        # (0001 1111) to see if it complies
        bitwise_reduction = bitwise_and(
            byte_array[big_endian_nth_byte['byte_2']],
            full(byte_arr_len, 31, dtype=uint8)
        )
        if count_nonzero(bitwise_reduction) > 0:
            return arr
        # next check the 4th byte for zeros
        if count_nonzero(byte_array[big_endian_nth_byte['byte_3']]) > 0:
            return arr
        # finally, check exponent. In 32-bit float, exponent can be +/- 128
        # in 16-bit float, exponent can be +/-16
        bitwise_exponent = frexp(not_nan)[1]
        if amax(bitwise_exponent) > 16 or amin(bitwise_exponent) < -16:
            return arr
        # else, return compressed
        return arr.astype(float16)

    return arr


def compress_array(arr: array, mode: str) -> CompressionResult:
    """
    compresses the array by finding the minimum value.
    if mode is 'e', the differences between elements are stored
    if mode is 'm', the returned array holds the difference from minimum
    :param arr: numpy source array
    :param mode: string, must be 'e' or 'm'. 'e' is differences between elements, 'm' is difference from minimum
    :return: CompressionResult named-tuple like numpy array, value. if mode='e', array has 1 fewer elements than arr
    and value is the starting value. If mode='m', value is minimum
    """
    if mode not in ['e', 'm']:
        raise CompressionModeInvalidError('Mode must be "e" or "m", {} found'.format(mode))

    if arr.dtype.kind not in ['f', 'u', 'i']:
        raise CompressionError('Could not compress. dtype kind {} not '
                               'eligible for compression.'.format(arr.dtype.kind))

    # if we're already tiny, no compression
    if arr.dtype.kind in ['u', 'i'] and arr.itemsize == 1:
        return arr
    if arr.dtype.kind == 'f' and arr.itemsize == 2:
        return arr

    # can't do an 'e' mode if size is 1
    if arr.size == 1 and mode == 'e':
        return arr

    reference_value = arr[0] if mode == 'e' else amin(arr)
    # else, continue on
    diff_array = None
    if mode == 'e':
        diff_array = ediff1d(arr)

    if mode == 'm':
        diff_array = arr - reference_value

    # calculate the size of data needed
    max_value = amax(diff_array)
    min_value = amin(diff_array)
    ret_array = None
    if diff_array.dtype.kind in ['u', 'i']:  # integer or unsigned integer
        if min_value < 0 and (-1 * min_value) > max_value:
            max_abs_value = -1 * min_value
        else:
            max_abs_value = max_value
        type_char = 'i' if min_value < 0 else 'u'
        if min_value < 0:
            num_bytes = determine_required_bytes_signed_integer(max_abs_value)
        else:
            num_bytes = determine_required_bytes_unsigned_integer(max_abs_value)
        ret_array = diff_array.astype(get_numpy_type(type_char, num_bytes * 8))
    if diff_array.dtype.kind == 'f':  # float
        # try to convert the array
        ret_array = compress_float_array(diff_array)
    return CompressionResult(ret_array, reference_value)


def decompress_array(arr: array, mode: str, reference_value) -> array:
    """
    Decodes a numpy array using a specified mode and reference value.
    :param arr: array to decompress
    :param mode: either 'e' for element-wise differences or 'm' for difference from minimum
    :param reference_value: first value of decompressed array if 'e', else the min value of the decompressed array
    :return: numpy array with decompressed data
    """
    if mode not in ['e', 'm']:
        raise CompressionModeInvalidError('Mode must be "e" or "m", {} found'.format(mode))

    if arr.dtype.kind not in ['f', 'u', 'i']:
        raise CompressionError('Could not compress. dtype kind {} not '
                               'eligible for compression.'.format(arr.dtype.kind))
    if mode == 'e':
        ret_array = cumsum(arr) + reference_value
        ret_array = insert(ret_array, 0, reference_value)
    elif mode == 'm':
        ret_array = add(arr, full(arr.shape, reference_value))
    return ret_array


def round_array_returning_integers(arr: array, num_decimals: int) -> array:
    """
    Multiplies the array by 10^num_decimals, rounds the array, and returns an integer array
    :param arr: source array
    :param num_decimals: number of decimals to keep
    :return: 64-bit integer array with rounded data
    """
    try:
        num_decimals = ensure_int(num_decimals)
    except NotIntegerException:
        raise NotIntegerException('Could not round array, parameter ''num_decimals'' was not an integer')
    if num_decimals < 0:
        raise ValueError('Could not round array, parameter ''num_decimals'' cannot be negative')
    rounded_array = arr * pow(10, num_decimals)
    rounded_array = around(rounded_array)
    return rounded_array.astype(int64)
