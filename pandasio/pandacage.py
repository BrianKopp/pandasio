from numpy import array, uint8, uint16, uint32
from typing import Union
from pandasio.pandabar import _PandaBar, NUM_BYTES_PER_DEFINITION_WITHOUT_IDENTIFIER
from pandasio.exceptions import DataWrongShapeError,\
    DataTypeNotSupportedError, CouldNotAcquireFileLockError
from pandasio.utils.binary import read_unsigned_int
from fcntl import flock, LOCK_EX, LOCK_SH, LOCK_UN, LOCK_NB
import time
import os


MAX_WRITE_BLOCK_WAIT_SECONDS = 60
MAX_READ_BLOCK_WAIT_SECONDS = 30


def utils_supported_kinds() -> list:
    return ['i', 'u', 'f']


class PandaCage:
    """
    Class wrapping around file format designed for pandas DataFrames.
    """
    def __init__(self, file_path=None):
        """
        Creates a new PandaCage object.
        :param file_path:
        """
        self.file_path = file_path
        self._timebox_version = 1
        self._bar_names_are_strings = True
        self._num_points = None
        self._num_bytes_for_identifier = None
        self._index_bars = {}  # like { identifier : PandaBar }
        self._bars = {}  # like { identifier : PandaBar }
        self._MAX_WRITE_BLOCK_WAIT_SECONDS = MAX_WRITE_BLOCK_WAIT_SECONDS
        self._MAX_READ_BLOCK_WAIT_SECONDS = MAX_READ_BLOCK_WAIT_SECONDS
        return

    def set_data(self, data: array, name: str, is_index: bool=False, bytes_per_value: int=None,
                 type_char: Union(int, str)=None):
        """
        Assigns data for one of the columns in the PandaCage. If not first column, must match the shape of the
        existing data
        :param data: numpy array containing data to set
        :param name: name for the data
        :param is_index: boolean indicating whether the column is an index
        :param bytes_per_value: number of bytes per value. if entered, numpy array will downcast
        :param type_char: integer or single character string describing which type of data to downcast to
        :return: None
        """
        if self._num_points is None:
            self._num_points = data.size
        elif data.size != self._num_points:
            raise DataWrongShapeError('data size did not match existing shape of PandaCage')
        if data.dtype.kind not in utils_supported_kinds():
            raise DataTypeNotSupportedError('The provided numpy data array had data type that is not supported')

        # if existing
        if is_index and name in self._index_bars:
            self._index_bars[name].set_data(data)
            return
        if is_index and name in self._bars:
            self._bars[name].set_data(data)
            return
        bar = _PandaBar(
            identifier=name,
            bytes_per_value=array.dtype.itemsize if bytes_per_value is None else bytes_per_value,
            type_char=array.dtype.kind if type_char is None else type_char,
            is_index=is_index,
            data=data
        )
        if is_index:
            self._index_bars[name] = bar
        else:
            self._bars[name] = bar
        return

    def get_data(self, name: str) -> array:
        """
        Retrieves the data identified by name
        :param name: string to lookup data
        :return: numpy array with the data
        """
        if name in self._index_bars:
            return self._index_bars[name]
        elif name in self._bars:
            return self._bars[name]
        raise KeyError('Could not find name {} in PandaCage'.format(name))

    def read(self):
        """
        This function reads the entire file contents into memory.
        Later it can be improved to only read certain tags
        :return: void
        """
        with self._get_fcntl_lock('r') as handle:
            try:
                # read in the data
                self._read_file_info(handle)
                self._read_bar_data(handle)
            finally:
                # release shared lock
                flock(handle, LOCK_UN)
        return

    def write(self):
        """
        writes the file out to file_name.
        requires an exclusive LOCK_EX fcntl lock.
        blocks until it can get a lock
        :return: void
        """
        # put a file in the same directory to block new shared requests
        # this prevents a popular file from blocking forever
        # note, this is a blocking function as it waits for other write events to finish
        file_is_new = not os.path.exists(self.file_path)
        with self._get_fcntl_lock('w') as handle:
            try:
                self._prepare_for_write()
                self._write_file_info(handle)
                self._write_bar_data(handle)
            except:  # TODO catch errors
                if file_is_new:
                    os.remove(self.file_path)
                raise Exception()
            finally:
                flock(handle, LOCK_UN)  # release lock
                block_file_name = self._blocking_file_name()
                if os.path.exists(block_file_name):
                    os.remove(block_file_name)
        return

    def _read_file_info(self, file_handle) -> int:
        """
        Reads the file info from a file_handle. Populates file internals
        :param file_handle: file handle object in 'rb' mode that is seeked to the correct position (0)
        :return: int, seek bytes increased since file_handle was received
        """
        self._timebox_version = read_unsigned_int(file_handle.read(1))
        self._decode_options(int(read_unsigned_int(file_handle.read(2))))
        num_bars = read_unsigned_int(file_handle.read(2))
        self._num_points = read_unsigned_int(file_handle.read(4))
        self._num_bytes_for_identifier = read_unsigned_int(file_handle.read(1))
        bytes_seek = 1 + 2 + 2 + 4 + 1

        bytes_for_bar_def = num_bars * (self._num_bytes_for_identifier + NUM_BYTES_PER_DEFINITION_WITHOUT_IDENTIFIER)
        bars = _PandaBar.decode_panda_bars_definitions_from_bytes(
            file_handle.read(bytes_for_bar_def),
            num_bytes_for_identifier=self._num_bytes_for_identifier,
            identifiers_are_strings=False
        )
        self._index_bars = dict([(i, b) for i, b in bars if b.is_index()])
        self._bars = dict([(i, b) for i, b in bars if not b.is_index()])
        bytes_seek += bytes_for_bar_def
        return bytes_seek

    def _write_file_info(self, file_handle) -> int:
        """
        Writes out the file info to the file handle
        :param file_handle: file handle object in 'wb' mode. pre-seeked to correct position (0)
        :return: int, seek bytes advanced in this method
        """
        array([uint8(self._timebox_version)], dtype=uint8).tofile(file_handle)
        array([uint16(self._encode_options())], dtype=uint16).tofile(file_handle)
        num_bars = len(self._index_bars) + len(self._bars)
        array([uint16(num_bars)], dtype=uint16).tofile(file_handle)
        array([uint32(self._num_points)], dtype=uint32).tofile(file_handle)

        self._update_required_bytes_for_tag_identifier()
        array([uint8(self._num_bytes_for_identifier)], dtype=uint8).tofile(file_handle)
        bytes_seek = 1 + 2 + 2 + 4 + 1

        index_bytes_list = [b.encode_info(self._num_bytes_for_identifier, True)
                            for _, b in self._index_bars]  # TODO make sure this is sorted
        index_bytes = b''.join([b.byte_code for b in index_bytes_list])
        index_num_bytes = sum([b.num_bytes for b in index_bytes_list])
        bars_bytes_list = [b.encode_info(self._num_bytes_for_identifier, True)
                           for _, b in self._bars]  # TODO make sure this is sorted
        bars_bytes = b''.join([b.byte_code for b in bars_bytes_list])
        bars_num_bytes = sum([b.num_bytes for b in bars_bytes_list])

        file_handle.write(index_bytes)
        file_handle.write(bars_bytes)
        bytes_seek += index_num_bytes + bars_num_bytes

        return bytes_seek

    def _prepare_for_write(self):
        """
        Performs sorting and compression to prepare for write
        :return: None
        """
        raise NotImplementedError()  # TODO

    def _validate_data_for_write(self):
        """
        This method checks the data to ensure that the data is good for write
        :return: void
        """
        data_sizes = [b.num_points() for b in self._index_bars.values()]
        data_sizes.extend([b.num_points() for b in self._bars.values()])
        if len([d for d in data_sizes if d != self._num_points]) > 0:
            raise DataWrongShapeError('Some data are not the right length')
        for b in self._index_bars.values():
            b.validate()
        for b in self._bars.values():
            b.validate()
        return

    def _write_bar_data(self, file_handle) -> int:
        """
        writes out the data
        :param file_handle: file handle object in 'wb' mode, pre-seeked to correct position
        :return: int, seek bytes advanced in this method
        """
        self._validate_data_for_write()
        seek_bytes = 0

        # then write out file data
        for b in self._index_bars:
            seek_bytes += self._index_bars[b].data_to_file(file_handle)
        for b in self._bars:
            seek_bytes += self._bars[b].data_to_file(file_handle)
        return seek_bytes

    def _read_bar_data(self, file_handle) -> int:
        """
        reads in data from the file handle
        :param file_handle: file handle in 'rb' mode, pre-seeked to the correct starting position
        :return: int, seek bytes advanced in this method
        """
        seek_bytes = 0
        for b in self._bars:
            seek_bytes += self._bars[b].data_from_file(file_handle, self._num_points)
        return seek_bytes

    def _decode_options(self, from_int: int):
        """
        Reads the options from the 1-byte options bit
        :param from_int: int holding options
        :return: void, populates class internals
        """
        # starting with the right-most bits and working left
        return

    def _encode_options(self) -> int:
        """
        Stores the bit-options in a 16-bit integer
        :return: int, no more than 16 bits
        """
        # note, this needs to be in the opposite order as _decode_options
        options = 0
        return options

    def _update_required_bytes_for_tag_identifier(self):
        """
        Looks at the tag list and determines what the max bytes required is
        :return: void, updates class internals
        """
        max_length = max([len(k) for k in self._tags])
        self._num_bytes_for_identifier = max_length * 4
        return

    def _blocking_file_name(self) -> str:
        """
        returns a file blocking name
        :return: file name of blocking file
        """
        return '{}.lock'.format(self.file_path)

    def _get_fcntl_lock(self, mode: str = 'r'):
        """
        gets a lock of type 'w' (writing) or 'r' (reading). throws error if can't get lock in time
        this is a blocking function, but doesn't block for more than the specified
        _MAX_READ/WRITE_BLOCK_WAIT_SECONDS.
        :param mode: single char, 'w' or 'r'
        :return: file handle if succeeded, raise exception if failed
        """
        if mode not in ['r', 'w']:
            raise ValueError('Could not get fcntl lock because mode specified was invalid: {}'.format(mode))
        block_file_name = self._blocking_file_name()
        count = 0
        sleep_seconds = 0.1
        file_locked = False
        handle = open(self.file_path, 'rb' if mode is 'r' else 'wb')
        if mode == 'r':
            # check and see if a blocking file exists, meaning we're waiting for a write job to clear
            # then try to get the lock
            while not file_locked and count <= (self._MAX_READ_BLOCK_WAIT_SECONDS / sleep_seconds):
                if not os.path.exists(block_file_name):
                    try:
                        flock(handle, LOCK_SH | LOCK_NB)
                        file_locked = True
                        break
                    except IOError:
                        pass
                count += 1
                time.sleep(sleep_seconds)
        if mode == 'w':
            block_file_is_mine = False
            while not file_locked and count <= (self._MAX_WRITE_BLOCK_WAIT_SECONDS / sleep_seconds):
                if not os.path.exists(block_file_name):
                    # put a blocking file
                    open(block_file_name, 'w').close()
                    block_file_is_mine = True
                elif not block_file_is_mine:  # file does exist, but it's not mine, wait patiently
                    time.sleep(sleep_seconds)
                else:  # file exists and it's mine
                    try:
                        flock(handle, LOCK_EX | LOCK_NB)
                        file_locked = True
                    except IOError:
                        time.sleep(sleep_seconds)
                        pass
                count += 1
            if not file_locked and block_file_is_mine and os.path.exists(block_file_name):
                os.remove(block_file_name)
        if not file_locked:
            handle.close()
            raise CouldNotAcquireFileLockError
        return handle