#!/bin/bash
set -e -v
echo "Compiling..."
javac *.java
echo "Running..."
java Main > prediction2.txt
