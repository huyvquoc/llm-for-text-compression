#!/bin/bash

TARGET_FOLDER="/data/hqvo3/weights/llama"
DATA_FOLDER="/data/hqvo3/courseworks/data_compression_info_theo/project/benchmarks/CanterburyCorpus"
COMPRESSION_FOLDER="./outputs/CanterburyCorpus_RZ"

# Specific binary files to process
BINARY_FILES=("cp.html" "ptt5" "sum" "kennedy.xls")

# Loop through the specific binary files
for filename in "${BINARY_FILES[@]}"; do
    file="$DATA_FOLDER/$filename"
    
    # Check if the file exists
    if [ -f "$file" ]; then
        # Create compression subfolder path
        compression_subfolder="$COMPRESSION_FOLDER/$filename"
        
        # Create the subfolder if it doesn't exist
        mkdir -p "$compression_subfolder"
        
        echo "Processing file: $filename"
        echo "Compression output: $compression_subfolder"
        
        # Run torchrun with RankZip compression
        CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 --master_port 29503 LLMzip_run.py \
            --ckpt_dir $TARGET_FOLDER/7B \
            --tokenizer_path $TARGET_FOLDER/tokenizer.model \
            --win_len 511 \
            --text_file "$file" \
            --compression_folder "$compression_subfolder" \
            --compression_alg "RankZip" \
            --encode_decode 0 \
            --batched_encode True
        
        echo "Completed: $filename"
        echo "-----------------------------------"
    else
        echo "Warning: File not found - $file"
        echo "-----------------------------------"
    fi
done

echo "All binary files processed!"
