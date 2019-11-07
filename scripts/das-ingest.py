#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
import h5py
import h5pyd
import time


################################################################################
def find_das_files(top_dir, sort=True):
    """Find all DAS-HDF5 files under ``top_dir``.

    Sort ascending in time if ``sort`` is ``True``. Return a list of DAS-HDF5
    files.
    """
    lggr.debug('Look for "*.h5" files under %s' % str(top_dir))
    h5_files = top_dir.glob('**/*.h5')
    das_files = list()
    lggr.debug('Check found HDF5 files')
    for h5f in h5_files:
        with h5py.File(str(h5f), 'r') as f:
            try:
                conv = f.attrs['Conventions']
            except Exception:
                continue

            if isinstance(conv, np.ndarray):
                if not np.any(conv == 'DAS-HDF5-1.0'):
                    continue
            else:
                if 'DAS-HDF5-1.0' not in conv:
                    continue

            t = f['t']
            das_files.append({'fname': str(h5f),
                              'tstart': t[0],
                              'tend': t[-1]})
    if len(das_files) == 0:
        lggr.debug('No HDF5 files found')
        return das_files
    lggr.debug('Found %d DAS-HDF5 files under %s' % (len(das_files), top_dir))
    if sort:
        lggr.debug('Sort DAS-HDF5 files in ascending time series order')
        das_files.sort(key=lambda d: d['tstart'])

    return [df['fname'] for df in das_files]


def transfer_dset(h5file, h5path, h5dom):
    """Transfer data and attributes of one HDF5 dataset"""
    lggr.debug('Transfer %r dataset data' % h5path)
    src_dset = h5file[h5path]
    if src_dset.shape == ():
        dset = h5dom.create_dataset(h5path, dtype=src_dset.dtype, shape=())
    else:
        dset = h5dom.create_dataset(h5path, dtype=src_dset.dtype,
                                    data=src_dset[...],
                                    fillvalue=src_dset.fillvalue)

    lggr.debug('Transfer %r dataset attributes' % h5path)
    for name, value in src_dset.attrs.items():
        if name not in ('CLASS', 'REFERENCE_LIST', 'DIMENSION_LIST'):
            dset.attrs[name] = value


def get_aggregation_size(das_files):
    """Find total data aggregation size by checking source DAS-HDF5 files"""
    lggr.debug('Calculate total aggregated data size in %d DAS-HDF5 files' %
               len(das_files))
    tot_samples = 0
    for das in das_files:
        with h5py.File(das, 'r') as f:
            tot_samples += f['trace'].shape[0]
    return tot_samples
################################################################################


parser = argparse.ArgumentParser(
    description='Aggregate DAS-HDF5 files into a Kita domain',
    epilog='Copyright (c) 2019 Akadio Inc.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('source', metavar='DIR', type=Path,
                    help='Top-level directory of DAS-HDF5 files to aggregate')
parser.add_argument('domain', metavar='DOMAIN', help=('Output Kita domain.'))
parser.add_argument('--bucket', metavar='BUCKET', help='S3 bucket name',
                    default='cordial-hsds')
parser.add_argument('--loglevel', default='info',
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help='Logging level')
parser.add_argument('--logto', metavar='FILE',
                    help='Log file (will overwrite if exists)')
arg = parser.parse_args()

if arg.logto:
    # Log to file
    logging.basicConfig(
        filename=arg.logto, filemode='w',
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=arg.loglevel.upper(),
        datefmt='%Y%m%dT%H%M%S')
else:
    # Log to stderr
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=arg.loglevel.upper(),
        datefmt='%Y%m%dT%H%M%S')
lggr = logging.getLogger('das-ingest')

lggr.debug('Top directory with input DAS-HDF5 files = %r' % arg.source)
lggr.debug('Output Kita domain = %r' % arg.domain)
lggr.debug('S3 bucket = %r' % arg.bucket)
lggr.debug('Logging level = %r' % arg.loglevel)
lggr.debug('Log to file = %r' % arg.logto)

if not arg.source.is_dir():
    raise OSError('%s: Not a directory or does not exist' % arg.source)
arg.source = arg.source.resolve()
lggr.info('Aggregate DAS-HDF5 files found under %s into Kita %r domain' %
          (arg.source, arg.domain))
das_files = find_das_files(arg.source)
lggr.debug('DAS-HDF5 files for processing: %r' % das_files)
lggr.info('Found %d DAS-HDF5 files for processing' % len(das_files))
if len(das_files) == 0:
    raise SystemExit('No DAS-HDF5 files found for processing')

lggr.info('Find the total size of the data agreggation')
tot_aggr_size = get_aggregation_size(das_files)
lggr.info('Total aggregated DAS data size = %d' % tot_aggr_size)

lggr.info('Open Kita domain %r in S3 bucket %r (will create if missing)' %
          (arg.domain, arg.bucket))
try:
    with h5pyd.File(arg.domain, mode='r', bucket=arg.bucket) as f:
        domain_exists = True
except Exception:
    domain_exists = False
domf = h5pyd.File(arg.domain, mode='a', bucket=arg.bucket)
das_valid_min = 1
das_valid_max = -1
first_das_file = True
for fname in das_files:
    lggr.info('Transfer from: %s' % fname)
    with h5py.File(fname, 'r') as h5f:
        if not domain_exists:
            lggr.info('New Kita domain, transfer some HDF5 content from the '
                      'DAS-HDF5 file')
            lggr.debug('Transfer root group attributes')
            dom_root = domf['/']
            for name, value in h5f['/'].attrs.items():
                if name not in ('time_coverage_end', 'source_checksum',
                                'source', 'date_created'):
                    dom_root.attrs[name] = value

            lggr.debug('Transfer HDF5 datasets: /crs, /channel, /x, /y, /z')
            transfer_dset(h5f, '/crs', domf)
            transfer_dset(h5f, '/channel', domf)
            channel = domf['/channel']
            channel.dims.create_scale(channel, 'channel')
            transfer_dset(h5f, '/x', domf)
            domf['/x'].dims[0].attach_scale(channel)
            transfer_dset(h5f, '/y', domf)
            domf['/y'].dims[0].attach_scale(channel)
            transfer_dset(h5f, '/z', domf)
            domf['/z'].dims[0].attach_scale(channel)

            trace_cursor = 0
            trace_valid_min = 1000
            trace_valid_max = 0
            domain_exists = True
            lggr.info('Done with new domain content transfer')

        if first_das_file:
            lggr.info(
                'First DAS-HDF5 file in this run')
            if 'trace' in domf['/'] and 'das' in domf['/'] and 't' in domf['/']:
                trace = domf['trace']
                das = domf['das']
                t = domf['t']
                if trace.shape[0] != das.shape[0] or t.shape[0] != das.shape[0]:
                    raise ValueError(
                        'Size mismatch between /trace, /das, and /t')
                trace_cursor = trace.shape[0]
                lggr.debug(
                    'Extend 1D HDF5 dataset /trace from %d to %d elements'
                    % (trace.shape[0], trace.shape[0] + tot_aggr_size))
                trace.resize(trace.shape[0] + tot_aggr_size, axis=0)

                lggr.debug(
                    'Extend 1D HDF5 dataset /das from %d to %d elements'
                    % (das.shape[0], das.shape[0] + tot_aggr_size))
                das.resize(das.shape[0] + tot_aggr_size, axis=0)

                lggr.debug(
                    'Extend 1D HDF5 dataset /t from %d to %d elements'
                    % (t.shape[0], t.shape[0] + tot_aggr_size))
                t.resize(t.shape[0] + tot_aggr_size, axis=0)

            elif ('trace' not in domf['/'] and 'das' not in domf['/'] and
                  't' not in domf['/']):
                trace_cursor = 0
                lggr.debug('Create new 1D HDF5 dataset /trace with init size %d'
                           % tot_aggr_size)
                trace = domf.create_dataset('trace', dtype=np.int64,
                                            shape=(tot_aggr_size,),
                                            maxshape=(None,),
                                            chunks=h5f['/trace'].chunks,
                                            fillvalue=-9999)
                trace.attrs['_FillValue'] = trace.fillvalue
                trace.attrs['long_name'] = h5f['trace'].attrs['long_name']
                trace.attrs['valid_min'] = h5f['trace'].attrs['valid_min']

                lggr.debug('Create new 1D HDF5 dataset /t with init size %d'
                           % tot_aggr_size)
                t = domf.create_dataset('t', dtype=np.float64,
                                        shape=(tot_aggr_size,),
                                        maxshape=(None,),
                                        chunks=h5f['/t'].chunks,
                                        fillvalue=np.nan)
                t.dims[0].attach_scale(trace)
                t.attrs['_FillValue'] = t.fillvalue
                t.attrs['standard_name'] = h5f['t'].attrs['standard_name']
                t.attrs['calendar'] = h5f['t'].attrs['calendar']
                t.attrs['units'] = h5f['t'].attrs['units']

                lggr.debug('Create new 1D HDF5 dataset /das with init size %d'
                           % tot_aggr_size)
                das = domf.create_dataset(
                    'das', dtype=h5f['das'].dtype,
                    shape=(tot_aggr_size, h5f['das'].shape[1]),
                    maxshape=(None, h5f['das'].shape[1]),
                    chunks=h5f['/das'].chunks,
                    fillvalue=np.nan, compression='gzip',
                    compression_opts=6)
                das.dims[0].attach_scale(trace)
                das.dims[1].attach_scale(channel)
                das.attrs['_FillValue'] = das.fillvalue
                das.attrs['long_name'] = h5f['das'].attrs['long_name']
                das.attrs['grid_mapping'] = h5f['das'].attrs['grid_mapping']
                das.attrs['coordinates'] = h5f['das'].attrs['coordinates']
                das.attrs['sampling_interval_seconds'] = \
                    h5f['das'].attrs['sampling_interval_seconds']

            else:
                raise RuntimeError(
                    'Not all of the /trace, /das, or /t datasets exists')

            trace_valid_min = trace.attrs['valid_min']
            trace_valid_max = h5f['trace'].attrs['valid_max']
            first_das_file = False

        lggr.info('Transfer data to /trace')
        src_trace = h5f['trace'][...]
        new_trace_vals = trace_cursor + np.arange(src_trace.shape[0])
        trace_cursor = new_trace_vals[-1] + 1
        for _ in range(5):
            try:
                trace[new_trace_vals[0]:trace_cursor] = new_trace_vals
                exc = None
            except (IOError, OSError) as e:
                lggr.warning('IOError/OSError exception caught, will sleep')
                exc = e
                time.sleep(1)
                continue
            else:
                break
        else:
            lggr.error('Max number of retries reached')
            if exc is not None:
                raise exc

        lggr.info('Transfer data to /t')
        src_t = h5f['t'][...]
        for _ in range(5):
            try:
                t[new_trace_vals[0]:trace_cursor] = src_t
                exc = None
            except (IOError, OSError) as e:
                lggr.warning('IOError/OSError exception caught, will sleep')
                exc = e
                time.sleep(1)
                continue
            else:
                break
        else:
            lggr.error('Max number of retries reached')
            if exc is not None:
                raise exc

        lggr.info('Transfer data to /das')
        src_das = h5f['das']
        for ch in range(0, src_das.shape[1], das.chunks[1]):
            lggr.info('das[%d:%d, %d:%d] = src_das[:, %d:%d]'
                      % (new_trace_vals[0], trace_cursor,
                         ch, ch + das.chunks[1],
                         ch, ch + das.chunks[1]))
            for _ in range(5):
                try:
                    das[new_trace_vals[0]:trace_cursor, ch:ch + das.chunks[1]] = \
                        src_das[:, ch:ch + das.chunks[1]]
                    exc = None
                except (IOError, OSError) as e:
                    lggr.warning('IOError/OSError exception caught, will sleep')
                    exc = e
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                lggr.error('Max number of retries reached')
                if exc is not None:
                    raise exc

        time_coverage_end = h5f.attrs['time_coverage_end']

        trace_valid_min = min(new_trace_vals.min(), trace_valid_min)
        trace_valid_max = max(new_trace_vals.max(), trace_valid_max)
        das_valid_min = min(src_das[...].min(), das_valid_min)
        das_valid_max = max(src_das[...].max(), das_valid_max)

# Add some attributes at the end...
domf.attrs['date_created'] = \
    datetime.utcnow().isoformat(timespec='seconds') + 'Z'
domf.attrs['time_coverage_end'] = time_coverage_end
das.attrs['valid_min'] = das_valid_min
das.attrs['valid_max'] = das_valid_max
trace.attrs['valid_min'] = trace_valid_min
trace.attrs['valid_max'] = trace_valid_max

domf.close()
lggr.info('Done')
