#!/bin/bash
set -e -v
echo "Compiling..."
javac -g *.java
echo "Running..."
java Main
