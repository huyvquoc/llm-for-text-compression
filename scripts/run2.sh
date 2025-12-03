#!/bin/bash

TARGET_FOLDER="/data/hqvo3/weights/llama"
DATA_FOLDER="/data/hqvo3/courseworks/data_compression_info_theo/project/benchmarks/CanterburyCorpus"
COMPRESSION_FOLDER="./outputs/CanterburyCorpus_RZ"

# Loop through all files in DATA_FOLDER
for file in "$DATA_FOLDER"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
        # Get the filename without path
        filename=$(basename "$file")
        
        # Create compression subfolder path
        compression_subfolder="$COMPRESSION_FOLDER/$filename"
        
        # Create the subfolder if it doesn't exist
        mkdir -p "$compression_subfolder"
        
        echo "Processing file: $filename"
        echo "Compression output: $compression_subfolder"
        
        # Run torchrun
        CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 --master_port 29501 LLMzip_run.py \
            --ckpt_dir $TARGET_FOLDER/7B \
            --tokenizer_path $TARGET_FOLDER/tokenizer.model \
            --win_len 511 \
            --text_file "$file" \
            --compression_folder "$compression_subfolder" \
            --compression_alg "RankZip"
        
        echo "Completed: $filename"
        echo "-----------------------------------"
    fi
done

echo "All files processed!