#!/bin/bash
set -e -v
echo "Compiling..."
javac *.java
echo "Running..."
java -Xmx3072m Main #> debug_spew.txt
