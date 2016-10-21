#!/bin/bash

# This script removes BOM (byte order mark) from files. This can be useful when
# some environments do not accept this encoding.
# See: github #7 for more details.

TMP_FILE=$(mktemp)

for f in $(find . -not -path '*/\.*' -type f \( ! -iname ".*" \)); do
    echo removing BOM from $f
    awk 'NR==1{sub(/^\xef\xbb\xbf/,"")}{print}' $f > $TMP_FILE && mv $TMP_FILE $f
done

rm -f $TMP_FILE
