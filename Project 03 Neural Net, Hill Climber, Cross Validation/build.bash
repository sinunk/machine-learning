#!/bin/bash
set -e -v
echo "Compiling..."
javac *.java
echo "Running..."
java Main #> project3.txt
