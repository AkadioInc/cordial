#!/bin/bash
sdir=$(dirname "${BASH_SOURCE[0]}")
# List all DAS-HDF5 files for processing...
s3ls=$(aws s3 ls --recursive s3://cordial1/DAS-HDF5/PoroTomo/ | cut -d " " -f 5)

# Repack the files...
for o in $s3ls; do echo $o; done | xargs -L 1 -P 3 -I {} ${sdir}/das-repack.sh {}
