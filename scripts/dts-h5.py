#!/usr/bin/env python3
import argparse
import logging
import numpy as np
from dateutil.parser import parse as dtparse
from datetime import datetime as dt
import h5py
from pyproj import Proj


################################################################################
def get_obs_times(line_str):
    """Convert a line with datetime strings to a list of datetime objects"""
    # Find the firs comma...
    lggr.debug('%r ...' % line_str[:67])
    first_comma = line_str.find(',')
    lggr.debug('First comma position = %d' % first_comma)
    times = line_str[first_comma + 1:].split(',')
    return [dtparse(dt) for dt in times]


def get_obs_data(file_, num_times):
    """Convert the remaining file's line to a NumPy array"""
    data = np.empty((0, num_times + 1), dtype=np.float32)
    for line in file_:
        line_data = np.fromstring(line, dtype=np.float32, sep=',')
        data = np.append(data, [line_data], axis=0)
    return data
################################################################################


parser = argparse.ArgumentParser(
    description='Transfer PoroTomo DTS data from CSV into an HDF5-based format',
    epilog='Copyright (c) 2019 Akadio Inc.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('fwd_csv', metavar='FILE',
                    help='CSV file with PoroTomo forward DTS data')
parser.add_argument('bckwd_csv', metavar='FILE',
                    help='CSV file with PoroTomo backward DTS data')
parser.add_argument('outfile', metavar='FILE', help='Output HDF5 file')
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
        datefmt='%Y-%m-%d %H:%M:%S')
else:
    # Log to stderr
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=arg.loglevel.upper(),
        datefmt='%Y-%m-%d %H:%M:%S')
lggr = logging.getLogger('das-to-h5')

lggr.debug('Forward DTS CSV file = %r' % arg.fwd_csv)
lggr.debug('Backward DTS CSV file = %r' % arg.bckwd_csv)
lggr.debug('Output HDF5 file = %r' % arg.outfile)
lggr.debug('Logging level = %r' % arg.loglevel)
lggr.debug('Log to file = %r' % arg.logto)

# Geolocation of the PoroTomo well with the DTS fiber-optic cable...
utm_x = 327961.538  # meters
utm_y = 4407554.942  # meters
elev_surface = 1230.67  # meters

# Epoch time...
epoch_time = '1970-01-01T00:00:00Z'
lggr.debug('Epoch time = %s' % epoch_time)
epoch_timestamp = dtparse(epoch_time).timestamp()
lggr.debug('Epoch timestamp = %f' % epoch_timestamp)


# Create the output HDF5 file and some of the root attributes...
lggr.info('Creating output DTS-HDF5 file %s (will overwrite)' % arg.outfile)
with h5py.File(arg.outfile, 'w') as h5:
    lggr.debug('Creating root attributes')
    h5.attrs['Conventions'] = 'DTS-HDF5-1.0, CF-1.7, ACDD-1.3'
    h5.attrs['references'] = 'https://gdr.openei.org/submissions/958'
    h5.attrs['project'] = 'CoRDIAL'
    h5.attrs['publisher_name'] = 'Akadio Inc.'
    h5.attrs['publisher_type'] = 'institution'
    h5.attrs['publisher_email'] = 'admin@akadio.com'
    h5.attrs['publisher_institution'] = 'Akadio Inc.'
    h5.attrs['publisher_url'] = 'https://www.akadio.com'
    h5.attrs['date_created'] = dt.utcnow().isoformat(timespec='seconds') + 'Z'
    h5.attrs['creator_name'] = 'Jeremy Patterson'
    h5.attrs['creator_type'] = 'person'
    h5.attrs['creator_institution'] = 'Uni. Wisconsin'
    h5.attrs['creator_email'] = 'jpatterson7@wisc.edu'
    h5.attrs['standard_name_vocabulary'] = \
        'CF Standard Names (v64, 5 March 2019)'
    h5.attrs['summary'] = ('These data are an 8-day time history of '
                           'vertical temperature measurements in Brady '
                           'observation well 56-1 collected during the '
                           'PoroTomo field experiment. The data was collected '
                           'with a fiber-optic DTS system installed to a depth '
                           'of 372 m below wellhead. DTS installation uses a '
                           'double-loop set up. Data includes forward length '
                           'and backward length temperature measurements.')
    h5.attrs['geospatial_lat_units'] = 'degree_north'
    h5.attrs['geospatial_lon_units'] = 'degree_east'
    utm11n = Proj(init='EPSG:32611')
    (h5.attrs['geospatial_lon_max'], h5.attrs['geospatial_lat_max']) = utm11n(
        utm_x, utm_y, inverse=True)
    (h5.attrs['geospatial_lon_min'], h5.attrs['geospatial_lat_min']) = (
        h5.attrs['geospatial_lon_max'], h5.attrs['geospatial_lat_max'])
    h5.attrs['keywords'] = ['PoroTomo', 'DTS', 'EGS', 'Brady', 'fiber-optics',
                            'temperature', 'Nevada', 'borehole',
                            'poroelastic tomography',
                            'distributed temperature sensing', 'hot springs']
    h5.attrs['license'] = \
        'Publicly accessible (https://creativecommons.org/licenses/by/4.0/)'

    # Coordinate reference system...
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

    # UTM easting...
    x = h5.create_dataset('x', data=utm_x, shape=())
    x.attrs['standard_name'] = 'projection_x_coordinate'
    x.attrs['long_name'] = 'Easting'
    x.attrs['units'] = 'm'

    # UTM northing...
    y = h5.create_dataset('y', data=utm_y, shape=())
    y.attrs['standard_name'] = 'projection_y_coordinate'
    y.attrs['long_name'] = 'Northing'
    y.attrs['units'] = 'm'

    # Forward DTS data...
    lggr.info('Reading forward DTS data from %r' % arg.fwd_csv)
    with open(arg.fwd_csv, 'rt') as f:
        # Read in measurement time data...
        lggr.debug('Getting measurement times...')
        times = get_obs_times(f.readline())
        lggr.debug('Found %d measurement times: %r ...' % (len(times),
                                                           times[:5]))
        # Skip a line
        f.readline()

        # Read the rest of the file's data...
        lggr.info('Reading the file data...')
        data = get_obs_data(f, len(times))
        lggr.info('Data size = %r' % (data.shape,))
    lggr.info('Done with forward DTS data file')

    lggr.info('Storing time coordinate...')
    t = h5.create_dataset(
        't', data=[dt.timestamp() - epoch_timestamp for dt in times],
        dtype=np.float32, fillvalue=np.nan)
    t.attrs['standard_name'] = 'time'
    t.attrs['calendar'] = 'gregorian'
    t.attrs['units'] = 'seconds since ' + epoch_time
    t.attrs.create('_FillValue', shape=(), data=t.fillvalue, dtype=t.dtype)
    t.dims.create_scale(t, 'time')

    lggr.info('Storing forward channel elevation...')
    elev = data[:, 0]
    data = data[:, 1:]  # drop the elevation for the array
    z = h5.create_dataset('/forward/z', data=elev_surface - elev,
                          fillvalue=np.nan)
    z.attrs['standard_name'] = 'height_above_reference_ellipsoid'
    z.attrs['long_name'] = 'Elevation'
    z.attrs['units'] = 'm'
    z.attrs['positive'] = 'up'
    z.attrs['surface_elevation'] = elev_surface
    z.attrs.create('valid_min', shape=(),
                   data=np.floor(np.nanmin(z[...])),
                   dtype=z.dtype)
    z.attrs.create('valid_max', shape=(),
                   data=np.ceil(np.nanmax(z[...])),
                   dtype=z.dtype)
    z.attrs.create('_FillValue', shape=(), data=z.fillvalue,
                   dtype=z.dtype)
    z.dims.create_scale(z, 'elevation')

    lggr.info('Storing forward DTS data...')
    # Transpose DTS data so it's shape is (time, elev)...
    data = data.T
    dts = h5.create_dataset('/forward/dts', data=data, fillvalue=np.nan)
    dts.attrs['long_name'] = 'depth temperature'
    dts.attrs['units'] = 'degC'
    dts.attrs['grid_mapping'] = crs.name
    dts.attrs['coordinates'] = ' '.join([t.name, z.name, x.name, y.name])
    dts.attrs.create('valid_min', shape=(), data=np.floor(np.nanmin(dts[...])),
                     dtype=dts.dtype)
    dts.attrs.create('valid_max', shape=(), data=np.ceil(np.nanmax(dts[...])),
                     dtype=dts.dtype)
    dts.attrs.create('_FillValue', shape=(), data=dts.fillvalue,
                     dtype=dts.dtype)
    dts.dims[0].attach_scale(t)
    dts.dims[1].attach_scale(z)

    # Backward DTS data...
    lggr.info('Reading backward DTS data from %r' % arg.bckwd_csv)
    with open(arg.bckwd_csv, 'rt') as f:
        # Read in measurement time data...
        lggr.debug('Getting measurement times...')
        times = get_obs_times(f.readline())
        lggr.debug('Found %d measurement times: %r ...' % (len(times),
                                                           times[:5]))
        # Skip a line
        f.readline()

        # Read the rest of the file's data...
        lggr.info('Reading the file data...')
        data = get_obs_data(f, len(times))
        lggr.info('Data size = %r' % (data.shape,))
    lggr.info('Done with backward DTS data file')

    # Compare the forward and backward times and store the latter if
    # different...
    bckwd_t = np.array([dt.timestamp() - epoch_timestamp for dt in times],
                       dtype=np.float32)
    if not np.array_equal(t[...], bckwd_t):
        lggr.debug('Max difference between forward and backward time = %f sec' %
                   np.max(np.abs(t[...] - bckwd_t)))
        lggr.info('Storing backward time coordinate...')
        t = h5.create_dataset(
            '/backward/t', data=bckwd_t, dtype=np.float32, fillvalue=np.nan)
        t.attrs['standard_name'] = 'time'
        t.attrs['calendar'] = 'gregorian'
        t.attrs['units'] = 'seconds since ' + epoch_time
        t.attrs.create('_FillValue', shape=(), data=t.fillvalue, dtype=t.dtype)
        t.dims.create_scale(t, 'time')

    lggr.info('Storing backward channel elevation...')
    elev = data[:, 0]
    data = data[:, 1:]  # drop the elevation for the array
    z = h5.create_dataset('/backward/z', data=elev_surface - elev,
                          fillvalue=np.nan)
    z.attrs['standard_name'] = 'height_above_reference_ellipsoid'
    z.attrs['long_name'] = 'Elevation'
    z.attrs['units'] = 'm'
    z.attrs['positive'] = 'up'
    z.attrs['surface_elevation'] = elev_surface
    z.attrs.create('valid_min', shape=(),
                   data=np.floor(np.nanmin(z[...])),
                   dtype=z.dtype)
    z.attrs.create('valid_max', shape=(),
                   data=np.ceil(np.nanmax(z[...])),
                   dtype=z.dtype)
    z.attrs.create('_FillValue', shape=(), data=z.fillvalue,
                   dtype=z.dtype)
    z.dims.create_scale(z, 'elevation')

    lggr.info('Storing backward DTS data...')
    # Transpose DTS data so it's shape is (time, elev)...
    data = data.T
    dts = h5.create_dataset('/backward/dts', data=data, fillvalue=np.nan)
    dts.attrs['long_name'] = 'depth temperature'
    dts.attrs['units'] = 'degC'
    dts.attrs['grid_mapping'] = crs.name
    dts.attrs['coordinates'] = ' '.join([t.name, z.name, x.name, y.name])
    dts.attrs.create('valid_min', shape=(), data=np.floor(np.nanmin(dts[...])),
                     dtype=dts.dtype)
    dts.attrs.create('valid_max', shape=(), data=np.ceil(np.nanmax(dts[...])),
                     dtype=dts.dtype)
    dts.attrs.create('_FillValue', shape=(), data=dts.fillvalue,
                     dtype=dts.dtype)
    dts.dims[0].attach_scale(t)
    dts.dims[1].attach_scale(z)

    # Add some more root group attributes now that all DTS data is known...
    h5.attrs['time_coverage_start'] = dt.fromtimestamp(
        np.min([t[...].min(), bckwd_t.min()])
    ).isoformat(timespec='seconds') + 'Z'
    h5.attrs['time_coverage_end'] = dt.fromtimestamp(
        np.max([t[...].max(), bckwd_t.max()])
    ).isoformat(timespec='seconds') + 'Z'

lggr.info('Done')
