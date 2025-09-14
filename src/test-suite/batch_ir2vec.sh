#!/bin/bash

# Script to run ir2vec on all .ll files in a directory
# Usage: ./batch_ir2vec.sh <folder_path> [output_file]

# Check if folder path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_path> [output_file]"
    echo "Example: $0 /path/to/llfiles results.txt"
    exit 1
fi

DEFS="$1"
if [[ "$DEFS" != "writeDefsMap" && "$DEFS" != "reachingDefsMap" ]]; then
    echo "Error: Invalid value for DEFS. Must be either 'writeDefsMap' or 'reachingDefsMap'"
    echo "Usage: $0 <writeDefsMap|reachingDefsMap>"
    exit 1
fi


FOLDER_PATH="$2"
OUTPUT_FILE="${3:-batch_results.txt}"

# Check if folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Directory '$FOLDER_PATH' does not exist."
    exit 1
fi

# Check if ir2vec executable exists
IR2VEC_PATH="./bin/ir2vec"
if [ ! -x "$IR2VEC_PATH" ]; then
    echo "Error: ir2vec executable not found at '$IR2VEC_PATH'"
    echo "Please make sure the executable exists and is executable."
    exit 1
fi

# Clear output file
> "$OUTPUT_FILE"

echo "Processing .ll files in: $FOLDER_PATH"
echo "Output will be written to: $OUTPUT_FILE"
echo "----------------------------------------"

# Initialize counters
count_ones=0
count_zeros=0
count_errors=0
count=0

# Process files and count results
while IFS= read -r -d '' ll_file; do
    count=$((count + 1))
    filename=$(basename "$ll_file")
    
    echo -ne "\rProcessing ($count): $filename"
    # Run ir2vec and capture the output
    output=$($IR2VEC_PATH -$DEFS -fa -level p -o test.txt "$ll_file" 2>&1)
    
    # Check if command was successful
    if [ $? -eq 0 ]; then
        # Extract the line we care about (should be the one with "Both maps")
        result_line=$(echo "$output" | grep "Both maps")
        
        if [ -n "$result_line" ]; then
            echo "$filename : $result_line" >> "$OUTPUT_FILE"
            
            # Get the last character of the result line
            last_char="${result_line: -1}"
            
            if [ "$last_char" = "1" ]; then
                count_ones=$((count_ones + 1))
            elif [ "$last_char" = "0" ]; then
                count_zeros=$((count_zeros + 1))
            else
                count_errors=$((count_errors + 1))
            fi
        else
            echo "$filename : No 'Both maps' output found" >> "$OUTPUT_FILE"
            count_errors=$((count_errors + 1))
        fi
    else
        echo "$filename : ERROR - $output" >> "$OUTPUT_FILE"
        count_errors=$((count_errors + 1))
    fi
    
    # Clean up temporary test.txt if it was created
    [ -f "test.txt" ] && rm -f "test.txt"
    
done < <(find "$FOLDER_PATH" -name "*.ll" -type f -print0 | sort -z)

echo "----------------------------------------"
echo "Processing complete! Results written to: $OUTPUT_FILE"
echo "Total files processed: $count"
echo ""
echo "Results Summary:"
echo "================"
echo "Files ending with 1: $count_ones"
echo "Files ending with 0: $count_zeros"
echo "Errors/No output: $count_errors"
