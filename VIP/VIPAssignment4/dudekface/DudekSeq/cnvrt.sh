#!/bin/sh -f
# Convert the format for a set of images
# Useage:  cnvrt files.
for i in $@ ; do
 # Extract the root, r, from the input filename, i
 r=`echo $i | sed -e '{ s/.TIF$//
     s/^.*\([0-9][0-9][0-9][0-9]\)$/frame\1/
      }'`  
 # Do the conversion from the input file type to the output file type
 tifftopnm $i > $r.ppm
 # Gzip the original (delete later)
 gzip $i
 # Convert colour to greylevel image
 ppmtopgm $r.ppm > $r.pgm
 # Ensure user/group read/write
 chmod ug+rw $r.ppm $r.pgm
 chmod o+r $r.ppm $r.pgm
 # compress pnm images
 # gzip $r.ppm $r.pgm
 echo "Done: "$r
done
