#!/bin/bash

set -o errexit
set -o pipefail

DIR_OF_SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXAMPLES_DIR="$DIR_OF_SCRIPT/../examples"

for i in $(ls $EXAMPLES_DIR); do
  if [[ -d "$EXAMPLES_DIR/$i" && $i =~ ^example* ]] ; then
    echo "Running example from $i"
    PYTHONPATH=$EXAMPLES_DIR/$i:$EXAMPLES_DIR:$DIR_OF_SCRIPT/..:$PYTHONPATH python -m main
  fi
done
