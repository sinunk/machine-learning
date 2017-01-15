#!/bin/bash
set -e -v
echo "Compiling..."
javac *.java
echo "Running..."
java Main > prediction3.txt
