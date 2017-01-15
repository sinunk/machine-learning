#!/bin/bash
set -e -v
echo "Compiling..."
javac *.java
echo "Running..."
java Main #> debug_27-11.txt