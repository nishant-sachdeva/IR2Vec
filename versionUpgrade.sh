#!/bin/bash

# Check if find-string and replace-string are provided as arguments
if [ $# -ne 2 ]; then
	echo "Usage: $0 <find-string> <replace-string>"
	exit 1
fi

find_string="$1"
replace_string="$2"

# Hardcoded list of filenames
filenames=(
	".devcontainer/devcontainer.json"
	".github/workflows/upload-pypi.yml"
	".github/workflows/wheel.yml"
	"Manylinux2014_Compliant_Source/pkg/regen-oracle.sh"
	"Manylinux2014_Compliant_Source/pkg/tests/test_ir2vec.py"
)

# Perform the replacement using sed for each file
for filename in "${filenames[@]}"; do
	if [ ! -f "$filename" ]; then
		echo "Error: File '$filename' not found. Skipping."
		continue
	fi

	sed -i "s/$find_string/$replace_string/g" "$filename"
	echo "Replacement complete. All occurrences of '$find_string' have been replaced with '$replace_string' in $filename."
done
