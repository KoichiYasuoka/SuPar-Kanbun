#! /bin/sh
for F
do split -a 2 -b 83886080 --numeric-suffixes=01 $F $F.
   ls -1 $F.0[1-9] | sed 's/^\(.*\)0\([1-9]\)$/mv & \1\2/' | sh
done
exit 0
