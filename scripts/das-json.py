import argparse
import json
import re
from pathlib import Path
import subprocess
import multiprocessing
import s3fs


def get_json(das_path):
    """Use ncks to generate JSON on a DAS-HDF5 file.

    Parameters
    ----------
    das_file : pathlib.Path
        A DAS-HDF5 file.

    Returns
    -------
    str
        ncks JSON output.
    """
    ncks = subprocess.run(['ncks', '-M', '-m', '--json', str(das_path)],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          encoding='utf-8', check=True)
    return ncks.stdout


def fix_json(nco_json, max_fixes=20):
    """Fix ncks-generated JSON to be valid.

    Parameters
    ----------
    nco_json : str
        ncks-generated JSON.
    max_fixes : int, optional
        The maximum number of fix attempts to make input JSON valid. Raises
        a RuntimeError exception when exceeded.

    Returns
    -------
    str
        The input JSON, now valid.
    """
    fix_count = 0
    while True:
        try:
            json.loads(nco_json)
        except json.JSONDecodeError as e:
            fix_count += 1
            if fix_count > 20:
                raise RuntimeError(
                    'Reached maximum number of JSON validation fixes')
            match = re.match(
                r"Expecting ',' delimiter: line \d+ column \d+ \(char (\d+)\)",
                str(e))
            if match:
                # Expected JSON decoding error, fix it...
                char_pos = int(match.group(1)) + 1
                nco_json = nco_json[:char_pos] + '0' + nco_json[char_pos:]
                continue
            else:
                # Unexpected JSON decoding error...
                raise e
        else:
            break
    return nco_json


def process_main(s3das):
    """Obtain NCO-JSON from a DAS-HDF5 file in S3.

    Download the DAS-HDF5 file, run ncks on it, save its JSON output in a file.

    Parameters
    ----------
    s3das : str
        The S3 path to a DAS-HDF5 file.
    """
    das_path = Path(Path(s3das).name)
    print(f'Downloading {s3das}')
    try:
        s3 = s3fs.S3FileSystem(anon=False)
        s3.get(s3das, str(das_path))
        nco_json = get_json(das_path)
        nco_json = fix_json(nco_json)
    except Exception as e:
        print(f'ERROR {s3das}')
        with open('ERROR_' + das_path.name, 'wt') as f:
            f.write(f'ERROR: {str(e)}')
        return
    else:
        print(f'Done {str(das_path)}')
        das_path.with_suffix('.json').write_text(nco_json)
    finally:
        das_path.unlink()


parser = argparse.ArgumentParser(
    description='Produce NCO JSON from DAS-HDF5 files',
    epilog='Copyright (c) 2019 Akadio Inc.')
parser.add_argument('s3path', help='S3 path that selects DAS-HDF5 files')
arg = parser.parse_args()

# List all DAS-HDF5 files to process...
s3 = s3fs.S3FileSystem(anon=False)
das_files = s3.ls(arg.s3path)
if len(das_files) == 0:
    raise SystemExit(f'{arg.s3path}: No DAS-HDF5 files')
else:
    print(f'{len(das_files)} DAS-HDF5 files to process.')

with multiprocessing.Pool(processes=3) as pool:
    pool.map(process_main, das_files)
