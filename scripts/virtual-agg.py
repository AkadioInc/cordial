#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
import s3fs
import os
import numpy as np
import h5py
import h5pyd
import time


################################################################################

def get_s3fs():
    """ Return a s3fs object for access to S3
    """
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        raise KeyError("AWS_ACCESS_KEY_ID environment not set")
    aws_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        raise KeyError("AWS_SECRET_ACCESS_KEY environment not set")
    aws_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    s3 = s3fs.S3FileSystem(key=aws_key_id, secret=aws_secret_key)
    return s3

def find_das_files(top_dir, sort=True, bucket=None):
    """Find all DAS-HDF5 files under ``top_dir``.

    Sort ascending in time if ``sort`` is ``True``. Return a list of DAS-HDF5
    files.
    """
    lggr.debug('Look for "*.h5" files under %s' % str(top_dir))
    s3 = get_s3fs()
    if bucket:
        s3path = bucket + top_dir
    else:
        s3path = top_dir
    # get sub-dirs of top_dir
    lggr.debug('Get files under %r' % s3path)
    dir_names = s3.ls(s3path)
    das_files = list()
    for dir_name in dir_names:
        lggr.debug('Get files under %r' % dir_name)
        items = s3.ls(dir_name)

        for item in items:
            if not item.endswith(".h5"):
                continue
            lggr.debug("Checking file %r" % item)
            with h5py.File(s3.open(item, 'rb')) as f:
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
                samples = f['trace'].shape[0]
                lggr.debug("Adding file %r samples %d" % (item, samples))
                das_files.append({'fname': item,
                              'tstart': t[0],
                              'tend': t[-1],
                              'samples': samples})
    if len(das_files) == 0:
        lggr.debug('No HDF5 files found')
        return das_files
    lggr.debug('Found %d DAS-HDF5 files under %s' % (len(das_files), top_dir))
    if sort:
        lggr.debug('Sort DAS-HDF5 files in ascending time series order')
        das_files.sort(key=lambda d: d['tstart'])

    return das_files


def transfer_dset(h5file, h5path, h5dom):
    """Transfer data and attributes of one HDF5 dataset"""
    lggr.debug('Transfer %r dataset data' % h5path)
    src_dset = h5file[h5path]
    dset = h5dom.create_dataset(h5path, dtype=src_dset.dtype,
                                    shape=src_dset.shape,
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
        tot_samples += das["samples"]
    return tot_samples
################################################################################


parser = argparse.ArgumentParser(
    description='Aggregate DAS-HDF5 files into a Kita domain',
    epilog='Copyright (c) 2019 Akadio Inc.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('source', metavar='DIR',
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

# changed source to a string, so no resolve - jlr
#arg.source = arg.source.resolve()
lggr.info('Aggregate DAS-HDF5 files found under %s into Kita %r domain' %
          (arg.source, arg.domain))
das_files = find_das_files(arg.source, bucket=arg.bucket)
#lggr.debug('DAS-HDF5 files for processing: %r' % das_files)

lggr.info('Found %d DAS-HDF5 files for processing' % len(das_files))
if len(das_files) == 0:
    raise SystemExit('No DAS-HDF5 files found for processing')

lggr.info('Find the total size of the data agreggation')
tot_aggr_size = get_aggregation_size(das_files)
lggr.info('Total aggregated DAS data size = %d' % tot_aggr_size)

lggr.info('Open Kita domain %r (will create if missing)' % arg.domain)
try:
    with h5pyd.File(arg.domain, mode='r') as f:
        domain_exists = True
except Exception:
    domain_exists = False
domf = h5pyd.File(arg.domain, mode='a')
das_valid_min = 1
das_valid_max = -1
first_das_file = True
s3 = get_s3fs()
dt_chunk_index = [("offset", np.int64), ("size", np.int32), ("file_uri", "S96")]
for das_file in das_files:
    fname = das_file["fname"]
    lggr.info('Transfer from: %s' % fname)
    with h5py.File(s3.open(fname, 'rb')) as h5f:
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
                                            chunks=(500_000,), fillvalue=-9999)
                trace.attrs['_FillValue'] = trace.fillvalue
                trace.attrs['long_name'] = h5f['trace'].attrs['long_name']
                trace.attrs['valid_min'] = h5f['trace'].attrs['valid_min']

                lggr.debug('Create new 1D HDF5 dataset /t with init size %d'
                           % tot_aggr_size)
                t = domf.create_dataset('t', dtype=np.float64,
                                        shape=(tot_aggr_size,),
                                        maxshape=(None,),
                                        chunks=trace.chunks,
                                        fillvalue=np.nan)
                t.dims[0].attach_scale(trace)
                t.attrs['_FillValue'] = t.fillvalue
                t.attrs['standard_name'] = h5f['t'].attrs['standard_name']
                t.attrs['calendar'] = h5f['t'].attrs['calendar']
                t.attrs['units'] = h5f['t'].attrs['units']

                das_chunks = h5f['das'].chunks
                lggr.info(f"das_chunks: {das_chunks}")
                das_shape = (tot_aggr_size, h5f['das'].shape[1])
                lggr.debug(f"das_shape: {das_shape}")

                # create dataset to store chunk locations
                # Note: assumes that the chunk shape is evenly divisible into das shape
                for i in range(2):
                    if h5f['das'].shape[i] % das_chunks[i] != 0:
                        lggr.error("das shape not divisible by chunk size")
                        raise ValueError("Invalid chunk shape")
                das_index_shape = (das_shape[0] // das_chunks[0], das_shape[1] // das_chunks[1])
                chunk_dset = domf.create_dataset(None, shape=das_index_shape, maxshape=(None, das_index_shape[1]), dtype=dt_chunk_index)
                lggr.info("chunk_dset id: %s", chunk_dset.id.id)
                chunks = {}
                chunks["class"] = 'H5D_CHUNKED_REF_INDIRECT'
                chunks["dims"] = das_chunks
                chunks["chunk_table"] = chunk_dset.id.id
                lggr.debug('Create new 1D HDF5 dataset /das with init size %d'
                           % tot_aggr_size)
                das = domf.create_dataset(
                    'das', dtype=h5f['das'].dtype,
                    shape=das_shape,
                    maxshape=(None, das_shape[1]),
                    chunks=chunks,
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
        ds = src_das.id
        num_chunks = ds.get_num_chunks()
        src_das_shape = src_das.shape
        bump = 0
        if src_das_shape[0] % das_chunks[0] != 0:
            lggr.warning(f"Das shape: {src_das_shape} dim 0 is not divisible by chunk shape for file: {fname}")
            bump = 1
        if src_das_shape[1] % das_chunks[1] != 0:
            msg = f"Das shape: {src_das_shape} dim 1 is not divisible by chunk shape for file: {fname}"
            lggr.error(msg)
            raise ValueError(msg)

        # create numpy array to store chunk info before writing to chunk_index dataset
        arr = np.zeros((src_das_shape[0] // das_chunks[0] + bump, src_das_shape[1] // das_chunks[1]), dtype=dt_chunk_index)
        s3path = "s3://" + fname
        lggr.info(f"das num chunks: {num_chunks}")
        lggr.debug("chunk info...")
        for i in range(num_chunks):
            chunk_info = ds.get_chunk_info(i)
            lggr.debug(f"{i}: {chunk_info}")
            e = (chunk_info.byte_offset, chunk_info.size, s3path)  # offset, size, file_uri
            arr[ chunk_info.chunk_offset[0] // das_chunks[0], chunk_info.chunk_offset[1] // das_chunks[1]] = e
        chunk_index_start = new_trace_vals[0] // das_chunks[0]
        chunk_index_end = chunk_index_start + arr.shape[0]
        lggr.debug(f"writing chunk_dset[{chunk_index_start}:{chunk_index_end}, :]")
        chunk_dset[chunk_index_start:chunk_index_end, :] = arr

        time_coverage_end = h5f.attrs['time_coverage_end']

        trace_valid_min = min(new_trace_vals.min(), trace_valid_min)
        trace_valid_max = max(new_trace_vals.max(), trace_valid_max)
        # TBD - how to efficiently determine this if we have not actually
        # reading the data?
        #das_valid_min = min(src_das[...].min(), das_valid_min)
        #das_valid_max = max(src_das[...].max(), das_valid_max)

# Add some attributes at the end...
domf.attrs['date_created'] = \
    datetime.utcnow().isoformat(timespec='seconds') + 'Z'
domf.attrs['time_coverage_end'] = time_coverage_end
#das.attrs['valid_min'] = das_valid_min
#das.attrs['valid_max'] = das_valid_max
trace.attrs['valid_min'] = trace_valid_min
trace.attrs['valid_max'] = trace_valid_max

domf.close()
lggr.info('Done')
