from warnings import warn
import logging
import numpy as np
import obspy
import h5py_switch
from .stream import DasStream


logging.getLogger(__name__)


class DasIo:
    """DAS measurements from one fiber-optic cable in the DAS-HDF5 format"""

    def __init__(self, path, mode, **kwargs):
        """Open DAS-HDF5 ``path`` for read access."""
        self._f = h5py_switch.File(path, mode, **kwargs)
        self._trace_ids = self._f['channel'][...]
        self._num_traces = self._f['channel'].shape[0]
        t = self._f['t']
        self._starttime = obspy.UTCDateTime(t[0])

    def __repr__(self):
        if self._f:
            return '<%s("%s", "%s") at 0x%x>' % (self.__class__.__name__,
                                                 self.filename, self._f.mode,
                                                 id(self))
        else:
            return '<%s(None) at 0x%x>' % (self.__class__.__name__, id(self))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the DasIo file."""
        self._f.close()
        self._f = None

    @property
    def filename(self):
        """DAS-HDF5 file name."""
        return self._f.filename

    @property
    def num_traces(self):
        """Number of DAS traces (channels)"""
        return self._num_traces

    @property
    def num_samples(self):
        """Number of time samples"""
        return self._f['t'].shape[0]

    @property
    def starttime(self):
        """DAS data start time as ObsPy UTCDateTime object"""
        return obspy.UTCDateTime(self._f['t'][0])

    @property
    def endtime(self):
        """DAS data end time as ObsPy UTCDateTime object"""
        return obspy.UTCDateTime(self._f['t'][-1])

    @property
    def trace_ids(self):
        """Trace (DAS channel) numeric identifiers as a NumPy array"""
        return self._trace_ids

    @property
    def instrument(self):
        """DAS instrument identifier"""
        return self._f.attrs['instrument']

    @property
    def sampling_rate_secs(self):
        """DAS data sampling rate in seconds"""
        return self._f['das'].attrs['sampling_interval_seconds']

    @property
    def sampling_rate_Hz(self):
        """DAS data sampling rate in Hz"""
        return 1. / self._f['das'].attrs['sampling_interval_seconds']

    def _select_trace(self, sel):
        """Array indices of the trace dimension coordinate based on selection.

        Parameters
        ----------
        sel : slice
            Describes the trace (DAS channel) ID values to select.

        Returns:
        numpy.ndarray
            Array indices of the HDF5 dataset holding trace (DAS channel) IDs.
        """
        start = max(sel.start or self.trace_ids[0], self.trace_ids[0])
        stop = min(sel.stop or self.trace_ids[-1], self.trace_ids[-1])
        if start > stop:
            raise ValueError(
                'Should selection start and stop values be reversed?')
        idx = np.nonzero((self.trace_ids >= start) & (self.trace_ids <= stop))
        return idx[0][::(sel.step or 1)]

    def _select_samples(self, sel):
        """Array indices of the time dimension coordinate based on selection.

        Parameters
        ----------
        sel : slice
            Describes the time range of trace (DAS channel) data to select.

        Returns:
        numpy.ndarray
            Array indices of the HDF5 dataset holding time of trace
            observations.
        """
        t = self._f['t']
        start = (sel.start or self._starttime).timestamp
        stop = (sel.stop or self._endtime).timestamp
        if start > stop:
            raise ValueError(
                'Should selection start and stop values be reversed?')

        # Figure out the array index of the dataset elements with the start and
        # stop values...
        start_idx = int(np.floor(
            (start - self.starttime.timestamp) / self.sampling_rate_secs))
        stop_idx = int(np.ceil(
            (stop - self.starttime.timestamp) / self.sampling_rate_secs))

        # Extend a little bit the (start_idx, stop_idx) range...
        start_idx = max(start_idx - 1000, 0)
        stop_idx = min(stop_idx + 1000, self.num_samples)

        data = t[start_idx:stop_idx]
        if data[0] > start or data[-1] < stop:
            raise ValueError('Failed to find start/end of selection range')
        idx = np.nonzero((data >= start) & (data <= stop))
        return start_idx + idx[0]

    def select(self, trace=None, time=None, stride=100):
        """Read in DAS channel data and return a DasStream object.

        The DAS data can be selected based on trace (channel) and time range.
        ``trace`` can be a Python slice object defining the range of fiber-optic
        channels to read data for.

        Similar with the ``time`` argument. It is a slice of ObsPy UTCDateTime
        objects defining the time interval of the DAS data. The step argument
        must be an integer and will be interpreted as seconds.

        ``stride`` defines how many DAS traces (channels) to read in one block.
        It is does not relate to the step parameter of the slice object.
        """
        if trace:
            if isinstance(trace, int):
                trace = slice(trace, trace)
            elif not isinstance(trace, slice):
                raise TypeError('"trace" argument is not a slice object: {}'
                                .format(type(trace)))
            elif trace.start and not isinstance(trace.start, int):
                raise TypeError('Trace start value is not an integer: {}'
                                .format(type(trace.start)))
            elif trace.stop and not isinstance(trace.stop, int):
                raise TypeError('Trace stop value is not an integer: {}'
                                .format(type(trace.stop)))
            elif trace.step and not isinstance(trace.step, int):
                raise TypeError('Trace step value is not an integer: {}'
                                .format(type(trace.step)))

        if time:
            if isinstance(time, str):
                # Consider this the upper limit of the selection range...
                time = obspy.UTCDateTime(time)
                time = slice(time - self.sampling_rate_secs,
                             time + self.sampling_rate_secs)
            elif isinstance(time, obspy.UTCDateTime):
                # Consider this the upper limit of the selection range...
                time = slice(time - self.sampling_rate_secs,
                             time + self.sampling_rate_secs)
            elif not isinstance(time, slice):
                raise TypeError('"time" argument is not a slice object: {}'
                                .format(type(time)))
            elif isinstance(time.start, str) and isinstance(time.stop, str):
                time = slice(obspy.UTCDateTime(time.start),
                             obspy.UTCDateTime(time.stop))
            elif time.start and not isinstance(time.start, obspy.UTCDateTime):
                raise TypeError(
                    '"time" start value is not an obspy.UTCDateTime: {}'
                    .format(type(time.start)))
            elif time.stop and not isinstance(time.stop, obspy.UTCDateTime):
                raise TypeError(
                    '"time" stop value is not an obspy.UTCDateTime: {}'
                    .format(type(time.stop)))
            elif time.step:
                warn('"time" step value is given but not used', RuntimeWarning)

        # DAS data selection based on its dimension coordinates...
        if trace:
            trace_idx = self._select_trace(trace)
        else:
            trace_idx = np.arange(self._num_traces)
        trace_idx = trace_idx.tolist()
        if time:
            time_idx = self._select_samples(time)
        else:
            time_idx = np.arange(self.num_samples)
        time_idx = time_idx.tolist()

        # Check if any of the selection slices yielded an empty array of
        # indices and return an empty DasStream object...
        if len(trace_idx) == 0 or len(time_idx) == 0:
            return DasStream()

        if len(time_idx) > 1_000_000:
            warn(f'Requesting {len(time_idx)} time samples', RuntimeWarning)

        # HDF5 dataset with DAS data...
        das = self._f['das']

        # Prepare ObsPy expected metadata for each trace (DAS channel)...
        metadata = {
            'sampling_rate': 1.0 / das.attrs['sampling_interval_seconds'],
            'npts': len(time_idx),
            'starttime': obspy.UTCDateTime(self._f['t'][time_idx[0]])}

        # Read in DAS data in blocks of stride traces at one time, then keep
        # only those traces that were selected...
        traces = list()
        t_slice = slice(time_idx[0], time_idx[-1] + 1, 1)
        actual_stride = min(stride, len(trace_idx))
        for idx in range(trace_idx[0], trace_idx[-1] + 1, actual_stride):
            ch_tseries = das[t_slice, idx:idx + stride]
            for i in range(0, ch_tseries.shape[1]):
                tr_idx = idx + i
                # if (trace.step or 1) != 1 and np.all(trace_idx != tr_idx):
                if tr_idx in trace_idx:
                    traces.append(
                        obspy.Trace(ch_tseries[:, i],
                                    header=dict(
                                        channel=str(self._trace_ids[tr_idx]),
                                        **metadata)))
        return DasStream(traces, trace_ids=self.trace_ids[trace_idx],
                         source=self.filename)

    def __getitem__(self, key):
        return self.select(time=key[0], trace=key[1])
