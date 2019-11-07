#!/bin/bash
# Change storage settings of several datasets in a DAS-HDF5 file.
# Specifically:
#
#   * Remove DEFLATE commpresion filter for /das dataset
#   * New chunk size of 2000x513 for /das
#   * New chunk size of 30000 for /t and /trace
echo "Processing $1"
outfile=$(basename $1)
aws s3 cp --quiet s3://cordial1/$1 /tmp/${outfile}
echo "Downloaded $1"
env H5TOOLS_BUFSIZE=500000000 h5repack --filter=/das:NONE \
                                       --layout=/t,/trace:CHUNK=30000 \
                                       --layout=/das:CHUNK=2000x513 \
                                       /tmp/${outfile} \
                                       ${PWD}/${outfile}

# Verify repacked HDF5 file...
h5ls ${PWD}/${outfile} > /dev/null 2>&1

if [[ $? -ne 0 ]]
then
    echo "ERROR: $1"
    touch "${PWD}/ERROR.${outfile}"
else
    echo "Repacked $1"
    aws s3 cp --quiet ${PWD}/${outfile} s3://cordial1/repack/$1
    echo "Uploaded repack/$1"
fi
rm -vf /tmp/${outfile} ${PWD}/${outfile}
echo "Done $1"
