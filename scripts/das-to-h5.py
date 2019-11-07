#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import re
from pathlib import Path
from dateutil.parser import parse
from datetime import datetime as dt
import hashlib
import h5py
import segyio
from pyproj import Proj


################################################################################
def get_files(top_dir):
    """Find all PoroTomo SEG-Y files starting from the ``top_dir``.

    Any file with the ".sgy" extension will be considered. ``top_dir`` is a
    pathlib.Path object.
    """
    return sorted(top_dir.glob('**/*.sgy'))


def header2dict(hdr):
    """Split SEG-Y file's header into a dict with "C" keys."""
    zz = [s.strip() for s in re.split('(C\d{2})', hdr)[1:]]
    return dict(zip(zz[:-1:2], zz[1::2]))


def header_fingerprint(f):
    """Extract several SEG-Y header fields from the file.

    These header fields will be used to verify that every other file belongs to
    the same measurement campaign and can go into the same HDF5 files.
    """
    hdr = header2dict(f.text[0].decode('ascii'))
    keep = {k: hdr[k] for k in ('C02', 'C05', 'C14', 'C15', 'C16')}
    lggr.debug('Fingerprint headers = %r' % keep)
    return keep


def from_same_campaign(file_hdr, check_hdr):
    """Compare a SEG-Y header from a file with the select header fields."""
    sel_fields = {k: file_hdr[k] for k in ('C02', 'C05', 'C14', 'C15', 'C16')}
    return check_hdr == sel_fields


def datetime_from_header(c09):
    """Extract trace start time from the C09 header field as datetime object.
    """
    match = re.match('UTC Timestamp of first sample: (.+)$', c09)
    dt_str = match.group(1) + 'Z'
    lggr.debug('Datetime string from C09 = %r' % dt_str)
    return parse(timestr=dt_str)


def sampling_interval(c16):
    """Extract sampling interval in µs from the C16 header field"""
    match = re.match('Sampling Interval \(us\): (.+)$', c16)
    return float(match.group(1))


def count_all_traces(segy_files):
    """Count the total number of traces from all SEG-Y files."""
    lggr.info('Counting traces from all input SEG-Y files...')
    total_traces = 0
    for segy in segy_files:
        with segyio.open(str(segy), mode='r', strict=False) as f:
            num_traces = f.trace.shape
        lggr.debug('%d traces in %s' % (num_traces, str(segy)))
        total_traces += num_traces
    lggr.info('Total number of traces from all SEG-Y files = %d' % total_traces)
    return total_traces


def read_geolocation(geo_file):
    """Read in DAS channel geolocation from the supplied file path."""
    geoloc = np.genfromtxt(arg.geo, dtype=np.dtype('float32'), skip_header=2,
                           delimiter=',', usecols=(0, 1, 2, 3),
                           names=('channel', 'x', 'y', 'z'))
    # Missing geolocation data is indicated with zeroes, replacing them with
    # NaN...
    missing_geo = np.where((geoloc['x'] == 0) | (geoloc['y'] == 0) |
                           (geoloc['z'] == 0))
    geoloc['x'][missing_geo] = np.nan
    geoloc['y'][missing_geo] = np.nan
    geoloc['z'][missing_geo] = np.nan

    return geoloc


def get_file_checksum(segy_file):
    """Compute SHA3-256 checksum of SEG-Y file."""
    cksum = hashlib.sha3_256()
    with segy_file.open('rb') as f:
        for chunk in iter(lambda: f.read(100_000_000), b''):
            cksum.update(chunk)
    return cksum.hexdigest()
################################################################################


parser = argparse.ArgumentParser(
    description='Convert PoroTomo DAS SEG-Y data into an HDF5-based format',
    epilog='Copyright (c) 2019 Akadio Inc.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('source', metavar='LOC', type=Path,
                    help='Source (file or directory) of PoroTomo SEG-Y data')
parser.add_argument('--output', '-o', metavar='LOC', type=Path,
                    default=Path('.').resolve(strict=True),
                    help=('Output HDF5 file. If LOC is a directory, each '
                          'SEG-Y file becomes a separate HDF5 file.'))
parser.add_argument('--geo', metavar='FILE',
                    help='CSV file with fiber cable channel geolocation data')
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
lggr = logging.getLogger('das-to-h5')

lggr.debug('PoroTomo SEG-Y data source = %r' % arg.source)
lggr.debug('Output HDF5 file = %r' % arg.output)
lggr.debug('Fiber-optic cable channel geolocation file  = %r' % arg.geo)
lggr.debug('Logging level = %r' % arg.loglevel)
lggr.debug('Log to file = %r' % arg.logto)

# Locate input PoroTomo SEG-Y files...
if arg.source.is_dir():
    lggr.info('Looking for PoroTomo SEG-Y files in the %s directory and below' %
              arg.source)
    segy_files = get_files(arg.source)
    lggr.debug('PoroTomo SEG-Y files = %r' % segy_files)
    lggr.info('Found %d SEG-Y files' % len(segy_files))
elif arg.source.is_file():
    lggr.info('PoroTomo SEG-Y file = %r' % arg.source)
    segy_files = [arg.source]
else:
    raise OSError('%r: Not a file or directory' % str(arg.source))

# Create the data structure connecting output HDF5 file name with input SEG-Y
# files...
if arg.output.is_dir():
    # Each SEG-Y file is translated into one HDF5 file...
    task = dict()
    for sgy in segy_files:
        task.update({arg.output.joinpath(sgy.with_suffix('.h5').name): [sgy]})
    total_traces = None
else:
    # Output is single HDF5 file so count the traces in all SEG-Y files...
    total_traces = count_all_traces(segy_files)

    task = {arg.output: segy_files}
lggr.debug('Processing task = %r' % task)

# Start the work...
for outfile, segy_files in task.items():
    lggr.info('Create HDF5 file %s (will overwrite)' % outfile)
    # Create the output HDF5 file and some of the root attributes...
    with h5py.File(outfile, 'w') as h5:
        h5.attrs['Conventions'] = ['DAS-HDF5-1.0', 'CF-1.7', 'ACDD-1.3']
        h5.attrs['project'] = 'CoRDIAL'
        h5.attrs['references'] = 'https://gdr.openei.org/submissions/980'
        h5.attrs['publisher_name'] = 'Akadio Inc.'
        h5.attrs['publisher_type'] = 'institution'
        h5.attrs['publisher_email'] = 'admin@akadio.com'
        h5.attrs['publisher_institution'] = 'Akadio Inc.'
        h5.attrs['publisher_url'] = 'https://www.akadio.com'
        h5.attrs['creator_name'] = 'Kurt Feigl'
        h5.attrs['creator_type'] = 'person'
        h5.attrs['creator_institution'] = 'Uni. Wisconsin'
        h5.attrs['creator_email'] = 'feigl@wisc.edu'
        h5.attrs['standard_name_vocabulary'] = \
            'CF Standard Names (v64, 5 March 2019)'
        h5.attrs['summary'] = ('These are raw data from the horizontal DAS '
                               '(DASH) deployment at the surface during '
                               'testing at the PoroTomo Natural Laboratory at '
                               'Brady Hot Spring in Nevada. Testing was '
                               'completed during March 2016.')
        h5.attrs['keywords'] = ['geothermal', 'PoroTomo', 'DAS', 'fiber optic',
                                'surface sensors', 'seismic array',
                                'distributed acoustic sensing',
                                'poroelastic tomography',
                                'Bradys geothermal field']
        h5.attrs['license'] = \
            'Publicly accessible (https://creativecommons.org/licenses/by/4.0/)'
        h5.attrs['acknowledgement'] = (
            'This material is based upon work supported by the U.S. '
            'Department of Energy, Office of Science, Office of Basic Energy '
            'Sciences under Award Number DE-SC0019654.')

        # Loop over SEG-Y files and translate their data to HDF5...
        first_segy_file = True
        trace_counter = 0
        segy_cksum = dict()
        for segy in segy_files:
            with segyio.open(str(segy), mode='r', strict=False) as f:
                # Figure out the number of time samples to store in HDF5 file...
                traces_in_h5 = total_traces if total_traces else f.trace.shape

                if first_segy_file:
                    lggr.info(
                        'First SEG-Y file in the sequence = %r' % str(segy))
                    # Collect SEG-Y header information from the first file which
                    # will be used to verify each of the found files belong to
                    # the same measurement campaign...
                    lggr.debug(
                        'Extract fingerprint "C" headers from %s' % str(segy))
                    hdr_fields = header_fingerprint(f)

                    # Get geolocation data if available...
                    if arg.geo:
                        lggr.info(
                            'Read fiber cable channel geolocation from %s' %
                            arg.geo)
                        geoloc = read_geolocation(arg.geo)

                        # Sanity check: same number of channels
                        if f.tracecount != geoloc.shape[0]:
                            raise ValueError(
                                'Fiber cable channel number mismatch')
                    else:
                        lggr.info('No fiber cable channel geolocation data')

                    # Create some HDF5 objects since this is a new HDF5 file...
                    lggr.debug(
                        'Creating additional HDF5 objects in output file')
                    h5.attrs['instrument'] = hdr_fields['C05']

                    # Channel dimension coordinate...
                    if arg.geo:
                        channel = h5.create_dataset('channel',
                                                    data=geoloc['channel'],
                                                    dtype=np.int32,
                                                    fillvalue=-9999)
                    else:
                        channel = h5.create_dataset(
                            'channel', data=np.arange(f.tracecount,
                                                      dtype=np.int32),
                            fillvalue=-9999)
                    channel.attrs['long_name'] = 'channel number'
                    channel.attrs.create('valid_min', shape=(),
                                         data=channel[...].min(),
                                         dtype=channel.dtype)
                    channel.attrs.create('valid_max', shape=(),
                                         data=channel[...].max(),
                                         dtype=channel.dtype)
                    channel.attrs.create('_FillValue', shape=(),
                                         data=channel.fillvalue,
                                         dtype=channel.dtype)
                    channel.dims.create_scale(channel, 'channel')

                    # Trace dimension coordinate...
                    trace = h5.create_dataset('trace', shape=(traces_in_h5,),
                                              data=np.arange(traces_in_h5),
                                              maxshape=(None,),
                                              chunks=(traces_in_h5,),
                                              fillvalue=-9999)
                    trace.attrs['long_name'] = 'trace number'
                    trace.attrs.create('valid_min', shape=(),
                                       data=trace[...].min(),
                                       dtype=trace.dtype)
                    trace.attrs.create('valid_max', shape=(),
                                       data=trace[...].max(),
                                       dtype=trace.dtype)
                    trace.attrs.create('_FillValue', shape=(),
                                       data=trace.fillvalue,
                                       dtype=trace.dtype)
                    trace.dims.create_scale(trace, 'trace')

                    # A dataset to hold coordinate reference system definition
                    # for the geolocation data...
                    crs = h5.create_dataset('crs', shape=(), dtype=np.uint8)
                    crs.attrs['grid_mapping_name'] = 'transverse_mercator'
                    crs.attrs['semi_major_axis'] = 6378137
                    crs.attrs['inverse_flattening'] = 298.257223563
                    crs.attrs['longitude_of_central_meridian'] = -117
                    crs.attrs['latitude_of_projection_origin'] = 0
                    crs.attrs['scale_factor_at_central_meridian'] = 0.9996
                    crs.attrs['false_easting'] = 500000
                    crs.attrs['false_northing'] = 0
                    crs.attrs['unit'] = 'metre'
                    crs.attrs['epsg_code'] = 'EPSG:32611'
                    crs.attrs['crs_wkt_uri'] = 'http://www.epsg-registry.org/export.htm?wkt=urn:ogc:def:crs:EPSG::32611'  # noqa
                    crs.attrs['crs_wkt'] = 'PROJCRS["WGS 84 / UTM zone 11N",BASEGEODCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1.0]]]],CONVERSION["UTM zone 11N",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.01745329252]],PARAMETER["Longitude of natural origin",-117,ANGLEUNIT["degree",0.01745329252]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1.0]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1.0]],PARAMETER["False northing",0,LENGTHUNIT["metre",1.0]]],CS[cartesian,2],AXIS["easting (E)",east,ORDER[1]],AXIS["northing (N)",north,ORDER[2]],LENGTHUNIT["metre",1.0],ID["EPSG",32611]]'  # noqa

                    # Time...
                    epoch_time = '1970-01-01T00:00:00Z'
                    epoch_timestamp = parse(epoch_time).timestamp()
                    lggr.debug('Epoch timestamp = %f' % epoch_timestamp)
                    t = h5.create_dataset('t', shape=trace.shape,
                                          dtype=np.float64,
                                          maxshape=trace.maxshape,
                                          chunks=trace.chunks, fillvalue=np.nan)
                    t.attrs['standard_name'] = 'time'
                    t.attrs['calendar'] = 'gregorian'
                    t.attrs['units'] = 'seconds since ' + epoch_time
                    t.attrs.create('_FillValue', shape=(), data=t.fillvalue,
                                   dtype=t.dtype)
                    t.dims[0].attach_scale(trace)

                    # Trace sampling interval in seconds...
                    delta_t = 1.0e-6 * sampling_interval(hdr_fields['C16'])
                    lggr.debug('Sampling interval = %f sec' % delta_t)

                    if arg.geo:
                        # UTM easting...
                        x = h5.create_dataset('x', data=geoloc['x'],
                                              fillvalue=np.nan)
                        x.attrs['standard_name'] = 'projection_x_coordinate'
                        x.attrs['long_name'] = 'Easting'
                        x.attrs['units'] = 'm'
                        x.attrs.create('valid_min', shape=(),
                                       data=np.floor(np.nanmin(x[...])),
                                       dtype=x.dtype)
                        x.attrs.create('valid_max', shape=(),
                                       data=np.ceil(np.nanmax(x[...])),
                                       dtype=x.dtype)
                        x.attrs.create('_FillValue', shape=(), data=x.fillvalue,
                                       dtype=x.dtype)
                        x.dims[0].attach_scale(channel)

                        # UTM northing...
                        y = h5.create_dataset('y', data=geoloc['y'],
                                              fillvalue=np.nan)
                        y.attrs['standard_name'] = 'projection_y_coordinate'
                        y.attrs['long_name'] = 'Northing'
                        y.attrs['units'] = 'm'
                        y.attrs.create('valid_min', shape=(),
                                       data=np.floor(np.nanmin(y[...])),
                                       dtype=y.dtype)
                        y.attrs.create('valid_max', shape=(),
                                       data=np.ceil(np.nanmax(y[...])),
                                       dtype=y.dtype)
                        y.attrs.create('_FillValue', shape=(), data=y.fillvalue,
                                       dtype=y.dtype)
                        y.dims[0].attach_scale(channel)

                        # Elevation...
                        z = h5.create_dataset('z', data=geoloc['z'],
                                              fillvalue=np.nan)
                        z.attrs['standard_name'] = \
                            'height_above_reference_ellipsoid'
                        z.attrs['long_name'] = 'Elevation'
                        z.attrs['units'] = 'm'
                        z.attrs.create('valid_min', shape=(),
                                       data=np.floor(np.nanmin(z[...])),
                                       dtype=z.dtype)
                        z.attrs.create('valid_max', shape=(),
                                       data=np.ceil(np.nanmax(z[...])),
                                       dtype=z.dtype)
                        z.attrs.create('_FillValue', shape=(), data=z.fillvalue,
                                       dtype=z.dtype)
                        z.dims[0].attach_scale(channel)

                        # Geospatial root attributes defining bounding box...
                        h5.attrs['geospatial_lat_units'] = 'degree_north'
                        h5.attrs['geospatial_lon_units'] = 'degree_east'
                        utm11n = Proj(init=crs.attrs['epsg_code'])
                        (h5.attrs['geospatial_lon_max'],
                         h5.attrs['geospatial_lat_max']) = utm11n(
                            x.attrs['valid_max'], y.attrs['valid_max'],
                            inverse=True)
                        (h5.attrs['geospatial_lon_min'],
                         h5.attrs['geospatial_lat_min']) = utm11n(
                            x.attrs['valid_min'], y.attrs['valid_min'],
                            inverse=True)

                    # This dataset holds the DAS data...
                    das = h5.create_dataset('das',
                                            dtype=np.float32,
                                            shape=(traces_in_h5, f.tracecount),
                                            maxshape=(None, f.tracecount),
                                            chunks=(2000, 513),
                                            fillvalue=np.nan)
                    das.attrs['long_name'] = 'strain'
                    das.attrs['grid_mapping'] = 'crs'
                    das.attrs.create('_FillValue', shape=(), data=das.fillvalue,
                                     dtype=das.dtype)
                    das.attrs['sampling_interval_seconds'] = delta_t
                    das.attrs['coordinates'] = 't'
                    if arg.geo:
                        das.attrs['coordinates'] += ' x y z'
                    das.dims[0].attach_scale(trace)
                    das.dims[1].attach_scale(channel)

                    # Done with first file stuff...
                    first_segy_file = False

                #
                # Transfer data from the SEG-Y file...
                #
                hdr = header2dict(f.text[0].decode('ascii'))
                if not from_same_campaign(hdr, hdr_fields):
                    raise RuntimeError(
                        '%s: Different header values' % str(segy))

                lggr.info('Transferring data from %s' % str(segy))

                # Trace start time...
                t0 = datetime_from_header(hdr['C09'])

                # Number of time samples in the SEG-Y file...
                num_traces = f.trace.shape

                # Time coordinate values in seconds since epoch...
                time = f.samples * delta_t + (t0.timestamp() - epoch_timestamp)
                t[trace_counter:trace_counter + num_traces] = time

                # DAS data...
                data = segyio.tools.collect(f.trace[:])
                if data.shape == (f.tracecount, num_traces):
                    data = data.T
                das[trace_counter:trace_counter + num_traces, :] = data

                # Update trace count...
                trace_counter += num_traces
                lggr.info('Transferred %d out of %d traces' %
                          (trace_counter, traces_in_h5))

            # Compute SEG-Y file checksum...
            cksum = get_file_checksum(segy)
            lggr.debug('%s checksum is %s' % (str(segy), cksum))
            segy_cksum[segy.name] = 'SHA3-256:' + cksum

            lggr.info('Done with %s' % str(segy))

        # Add some more root group attributes now that all DAS data is known...
        h5.attrs['date_created'] = \
            dt.utcnow().isoformat(timespec='seconds') + 'Z'
        h5.attrs['time_coverage_start'] = dt.fromtimestamp(
            t[...].min()).isoformat(timespec='microseconds') + 'Z'
        h5.attrs['time_coverage_end'] = dt.fromtimestamp(
            t[...].max()).isoformat(timespec='microseconds') + 'Z'
        h5.attrs['source'] = np.string_(list(segy_cksum.keys()))
        h5.attrs['source_checksum'] = np.string_(list(segy_cksum.values()))

lggr.info('Done')
