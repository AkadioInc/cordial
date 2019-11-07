import logging
import numpy as np
import obspy


logging.getLogger(__name__)


class DasStream(obspy.Stream):
    """A collection of trace (DAS channel) time series.

    Inherits from ``obspy.Stream``.
    """

    def __init__(self, traces, trace_ids=None, source=None):
        super(DasStream, self).__init__(traces)
        if trace_ids is None:
            self._trace_ids = np.array([], dtype='i4')
        else:
            self._trace_ids = trace_ids
        self._source = source

    def __str__(self, extended=False):
        if self._source:
            msg = '%s from "%s" at 0x%x\n' % (self.__class__.__name__,
                                              self._source,
                                              id(self))
        else:
            msg = ''
        msg += super().__str__()
        msg = msg[:msg.rfind('\n')]
        return msg

    @property
    def trace_ids(self):
        """Trace (DAS channel) numeric identifiers as a NumPy array."""
        return self._trace_ids

    @property
    def source(self):
        """Source (file) of the DAS stream."""
        return self._source
