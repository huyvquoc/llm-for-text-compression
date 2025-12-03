# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# ============================================================================
# LLMzip: Text Compression using Large Language Models
# ============================================================================
# This script implements text compression and decompression using LLaMA models.
# It supports two compression algorithms:
#   1. Arithmetic Coding (AC) - entropy-based compression
#   2. RankZip (RZ) - rank-based compression
# The LLM provides probability distributions for efficient compression.

from typing import Tuple
import os
import sys
import torch  # PyTorch for deep learning operations
import fire  # Command-line interface generator
import time  # For timing operations
import json  # For reading/writing metrics
import numpy as np  # For numerical operations

from pathlib import Path

# FairScale for model parallelism across multiple GPUs
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

# LLaMA model components and LLMzip compression/decompression classes
from llama import ModelArgs, Transformer, Tokenizer, LLMzip_encode, LLMzip_decode

### Command to run
# torchrun --nproc_per_node 1 LLMzip_run.py --ckpt_dir weights/7B --tokenizer_path weights/tokenizer.model 
# --win_len 511 --text_file *.txt --compression_folder LLMzip_compression   > Log_files/text8_ent1.txt 2>&1

### For precise reproduction of the paper results set the following options
# compression_alg - 'both', encode_decode - 0, batched_encode = True, verify_save_decoded = 0, with_context_start = True

def setup_model_parallel() -> Tuple[int, int]:
    """
    Initialize distributed training environment for model parallelism.
    
    Sets up PyTorch distributed process group using NCCL backend for GPU communication.
    Initializes FairScale's model parallel framework to split the model across GPUs.
    
    Returns:
        Tuple[int, int]: (local_rank, world_size)
            - local_rank: GPU rank for this process (0 to world_size-1)
            - world_size: Total number of GPUs participating in parallelism
    """
    # Get distributed environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))  # Current GPU ID
    world_size = int(os.environ.get("WORLD_SIZE", -1))  # Total number of GPUs
    print("Local Rank : ",local_rank,", World Size : ",world_size)

    # Initialize NCCL backend for efficient GPU-to-GPU communication
    torch.distributed.init_process_group("nccl")
    # Set up model parallelism with FairScale
    initialize_model_parallel(world_size)
    # Assign this process to its designated GPU
    torch.cuda.set_device(local_rank)

    # Ensure reproducibility: seed must be the same across all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int
):
    """
    Load LLaMA model checkpoint and create encoder/decoder instances.
    
    Args:
        ckpt_dir: Directory containing model checkpoint files (*.pth) and params.json
        tokenizer_path: Path to the SentencePiece tokenizer model file
        local_rank: GPU rank for loading the corresponding checkpoint shard
        world_size: Total number of GPUs (must match number of checkpoint shards)
        max_seq_len: Maximum sequence length the model can process
        max_batch_size: Maximum batch size for inference
    
    Returns:
        Tuple[LLMzip_encode, LLMzip_decode]: Encoder and decoder instances
            wrapping the loaded LLaMA model for compression/decompression
    """
    start_time = time.time()
    
    # Find all checkpoint shards in the directory
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # Verify that number of checkpoint files matches expected parallelism
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # Each GPU loads its corresponding checkpoint shard
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Load model hyperparameters from JSON configuration file
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    # Initialize model arguments with sequence length, batch size, and loaded params
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    # Load tokenizer and set vocabulary size
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    
    # Use half-precision (FP16) for memory efficiency during model initialization
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)  # Reset to FP32
    
    # Load pretrained weights (strict=False allows partial loading)
    model.load_state_dict(checkpoint, strict=False)
    
    # Wrap model in encoder/decoder classes for compression operations
    Encoder = LLMzip_encode(model, tokenizer)
    Decoder = LLMzip_decode(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return Encoder,Decoder

def verify_text(compressed_file_name,text_file,text_decoded,context_txt,save_decoded,alg):
    """
    Verify decoded text against original and calculate quality metrics.
    
    Compares the decompressed text with the original to check if compression
    was lossless. Computes PSNR metric and optionally saves decoded output.
    
    Args:
        compressed_file_name: Base filename for saving decoded output
        text_file: Path to original input text file
        text_decoded: Decompressed text string
        context_txt: Initial context to exclude from comparison (if used)
        save_decoded: Whether to save the decoded text to file
        alg: Compression algorithm name ('ArithmeticCoding' or 'RankZip')
    
    Returns:
        float: PSNR value (Peak Signal-to-Noise Ratio) in dB
    """
    # Read original file (handle both text and binary files)
    try:
        # First try to read as UTF-8 text
        with open(text_file, 'r', encoding='utf-8') as txt_enc:
            text_encoded = txt_enc.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, read as binary and decode with latin-1 (preserves all bytes)
        with open(text_file, 'rb') as txt_enc:
            text_encoded = txt_enc.read().decode('latin-1')

    # Remove initial context from comparison if it was provided at decode time
    if context_txt is not None:
        text_encoded = text_encoded[len(context_txt):]
        text_decoded = text_decoded[len(context_txt):]

    # Check if compression is lossless (exact match)
    print(f'Using {alg}...')
    if text_encoded == text_decoded:
        print('[LOSSLESS]: The decoded text is the same as the original text')
    else:
        print('[LOSSY]: The decoded text is not the same as the original text')

    # Calculate PSNR metric to quantify reconstruction quality
    print('==== PSNR ====')
    psnr_val = calculate_psnr(text_encoded, text_decoded)
    print('=> psnr value: ', psnr_val)

    # Optionally save decoded text for inspection
    if save_decoded:
        if alg == 'ArithmeticCoding':
            with open(compressed_file_name+'_AC_decoded_text.txt','w') as txt_dec:
                txt_dec.write(text_decoded)
        else:
            with open(compressed_file_name+'_RZ_decoded_text.txt','w') as txt_dec:
                txt_dec.write(text_decoded )

    return psnr_val

def calculate_psnr(original_text, reconstructed_text):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between original and reconstructed text.
    
    Treats text as byte sequences and calculates Mean Squared Error (MSE).
    Higher PSNR indicates better reconstruction quality.
    PSNR = 20 * log10(MAX_VALUE / sqrt(MSE))
    
    Args:
        original_text: Original text string
        reconstructed_text: Reconstructed/decoded text string
    
    Returns:
        float: PSNR value in dB, or inf if texts are identical (perfect reconstruction)
    """
    # Convert text strings to UTF-8 byte arrays for numerical comparison
    original_bytes = np.frombuffer(original_text.encode('utf-8'), dtype=np.uint8)
    reconstructed_bytes = np.frombuffer(reconstructed_text.encode('utf-8'), dtype=np.uint8)
    
    # Pad shorter sequence with zeros to ensure equal length for comparison
    max_len = max(len(original_bytes), len(reconstructed_bytes))
    original_padded = np.pad(original_bytes, (0, max_len - len(original_bytes)), 'constant')
    reconstructed_padded = np.pad(reconstructed_bytes, (0, max_len - len(reconstructed_bytes)), 'constant')
    
    # Calculate Mean Squared Error between byte sequences
    mse = np.mean((original_padded.astype(np.float64) - reconstructed_padded.astype(np.float64)) ** 2)
    
    # Return infinity for perfect reconstruction (no error)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    
    # Maximum possible byte value (8-bit unsigned)
    max_value = 255.0
    
    # Calculate PSNR using standard formula
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    
    return psnr
        
# def write_dict_to_xlsx(data_dict, output_filename='output.xlsx', sheet_name='Sheet1'):
#     """
#     Write a dictionary to an Excel file using pandas.
    
#     Args:
#         data_dict: Dictionary with key-value pairs to write
#         output_filename: Name of the output Excel file (default: 'output.xlsx')
#         sheet_name: Name of the Excel sheet (default: 'Sheet1')
    
#     Returns:
#         None
#     """
#     import pandas as pd
    
#     # Convert dictionary to DataFrame
#     # If values are lists, pandas will create columns with those lists as values
#     # If values are scalars, it will create a single-row DataFrame
#     df = pd.DataFrame(data_dict)
    
#     # Write to Excel file
#     df.to_excel(output_filename, sheet_name=sheet_name, index=False)
    
#     print(f'Data successfully written to {output_filename}')

def write_dict_to_csv(data_dict, output_filename='output.csv'):
    """
    Write a dictionary to a CSV file using pandas.
    
    Converts dictionary to OrderedDict to preserve column order, then
    writes to CSV format. Used for saving compression metrics.
    
    Args:
        data_dict: Dictionary with key-value pairs (metrics to save)
        output_filename: Name of the output CSV file (default: 'output.csv')
    
    Returns:
        None
    """
    import pandas as pd
    from collections import OrderedDict
    
    # Convert to OrderedDict to preserve column order in CSV
    ordered_data = OrderedDict(data_dict)
    
    # Convert OrderedDict to pandas DataFrame for easy CSV writing
    df = pd.DataFrame(ordered_data)
    
    # Write to CSV file without row indices
    df.to_csv(output_filename, index=False)
    
    print(f'Data successfully written to {output_filename}')
    
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    win_len: int,
    text_file: str, 
    compression_folder: str,
    max_seq_len: int = 512,
    max_batch_size: int = 64,
    compression_alg: str = 'ArithmeticCoding',
    encode_decode: int = 2,
    batched_encode: bool = False,
    verify_save_decoded: int = 2,
    with_context_start: bool = False
):
    """
    Main function for LLMzip compression and decompression.
    
    Performs text compression using LLaMA language model with either
    Arithmetic Coding or RankZip algorithm. Can encode, decode, or both.
    
    Args:
        ckpt_dir: Directory containing LLaMA model checkpoint files
        tokenizer_path: Path to SentencePiece tokenizer model
        win_len: Context window length (must be <= max_seq_len, typically 511)
        text_file: Path to input text file to compress
        compression_folder: Output directory for compressed files and metrics
        max_seq_len: Maximum sequence length (default: 512)
        max_batch_size: Maximum batch size for inference (default: 32)
        compression_alg: Algorithm to use:
            - 'ArithmeticCoding': Entropy-based compression
            - 'RankZip': Rank-based compression
            - 'both': Run both algorithms
        encode_decode: Operation mode:
            - 0: Only encode (compress)
            - 1: Only decode (decompress)
            - 2: Both encode and decode (default)
        batched_encode: Use batched encoding for faster compression
            (WARNING: decoding doesn't work with batched mode)
        verify_save_decoded: Verification level:
            - 0: Don't verify or save decoded text
            - 1: Only verify (calculate PSNR)
            - 2: Verify and save decoded text (default)
        with_context_start: If True, save initial context separately
            and provide it at decode time (avoids encoding initial tokens)
    
    Returns:
        None (outputs saved to compression_folder)
    """

    # ========================================================================
    # Parameter Validation
    # ========================================================================
    # Validate that window length doesn't exceed model's maximum sequence length
    assert win_len <= max_seq_len, f'Window length {win_len} is greater than {max_seq_len}'
    # Validate encode/decode mode is one of the three valid options
    assert encode_decode in [0,1,2], f'encode_decode not in {[0,1,2]}'
    # Validate compression algorithm selection
    assert compression_alg in ['ArithmeticCoding','RankZip','both'], 'compression_alg not one of ArithmeticCoding / RankZip / both'

    # Warning for incompatible options
    if batched_encode:
        print("Warning decoding doesn't work when using batched encode")

    # ========================================================================
    # Setup and Initialization
    # ========================================================================
    start_time_main = time.time()
    # Initialize distributed environment
    local_rank, world_size = setup_model_parallel()
    # Suppress output from non-primary GPUs to avoid duplicate logs
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    
    # Convert encode_decode integer to boolean flags
    encode = encode_decode%2 == 0  # True if encode_decode is 0 or 2
    decode = encode_decode>0       # True if encode_decode is 1 or 2
    
    # Disable batched encoding if decoding is required (incompatible)
    if decode:
        batched_encode = False 

    # Load LLaMA model and create encoder/decoder wrappers
    Encoder,Decoder = load( ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)

    # Create output directory if it doesn't exist
    os.makedirs(compression_folder,exist_ok=True)
    # Base filename for all output files (includes window length)
    compressed_file_name = compression_folder + f'/LLMzip_{win_len}' 

    # Read input file to be compressed (handle both text and binary files)
    try:
        # First try to read as UTF-8 text
        with open(text_file, 'r', encoding='utf-8') as f_in:
            text_input = f_in.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, read as binary and decode with latin-1 (preserves all bytes)
        with open(text_file, 'rb') as f_in:
            text_input = f_in.read().decode('latin-1')

    # ========================================================================
    # ENCODING (Compression) Phase
    # ========================================================================
    if encode:
        # Tokenize input text (convert text to integer token IDs)
        # bos=False, eos=False: don't add special beginning/end tokens
        tokens_full = np.array(Encoder.tokenizer.encode(text_input,bos=False,eos=False))

        # If using context start, save initial tokens separately
        # This allows decoder to skip encoding these tokens
        if with_context_start:
            starter_tokens = tokens_full[:win_len]
            np.save(compressed_file_name+'_starter_tokens.npy',starter_tokens)

        # Alternative: encode only a specific portion for consistent comparisons
        # tokens_full = np.array(Encoder.tokenizer.encode(text_input,bos=False,eos=False))[511-win_len:]

        # Perform compression using specified algorithm
        # Returns dictionary with compression metrics (entropy, file size, etc.)
        ret_metric_dict = Encoder.encode_from_tokens(win_len,compression_alg,compressed_file_name,tokens_full=tokens_full,batched_encode=batched_encode,with_context_start=with_context_start)
    

    # ========================================================================
    # DECODING (Decompression) Phase
    # ========================================================================
    if decode:
        # Load number of tokens from saved metrics (needed for decoding)
        with open(compressed_file_name+'_metrics.json') as metrics_file:
            total_length = json.load(metrics_file)['N_T'][0]  # Total tokens to decode
        
        # Load initial context if it was saved during encoding
        if with_context_start:
            starter_tokens = np.load(compressed_file_name+'_starter_tokens.npy')
            # Convert starter tokens back to text for verification
            context_txt = Encoder.tokenizer.decode(starter_tokens.tolist())
        else:
            starter_tokens = None
            context_txt = None

        # --- Arithmetic Coding Decompression ---
        if (compression_alg == 'ArithmeticCoding')or(compression_alg =='both'): 
            compressed_file_name_full = compressed_file_name+'_AC.txt'
            
            # Decode the arithmetic coded bitstream
            decoded_text_ac = Decoder.decode_AC(win_len,starter_tokens,total_length, compressed_file_name_full)
            
            # Verify decoded text matches original and calculate PSNR
            if verify_save_decoded > 0:
                psnr_val = verify_text(compressed_file_name,text_file,decoded_text_ac,context_txt,verify_save_decoded==2,'ArithmeticCoding')

            ret_metric_dict['PSNR'] = psnr_val
            
        # --- RankZip Decompression ---
        if (compression_alg == 'RankZip')or(compression_alg =='both'): 
            compressed_file_name_full = compressed_file_name+'_RZ.txt'
            decompressed_file_name = compressed_file_name+'_RZ'

            # Decode the rank-encoded file
            decoded_text_rz = Decoder.decode_ranks(win_len,starter_tokens, compressed_file_name_full)
            
            # Verify decoded text matches original and calculate PSNR
            if verify_save_decoded > 0:
                psnr_val = verify_text(compressed_file_name,text_file,decoded_text_rz,context_txt,verify_save_decoded==2,'RankZip')

            ret_metric_dict['PSNR'] = psnr_val

    # ========================================================================
    # Save Results and Metrics
    # ========================================================================
    # Calculate total execution time
    total_processed_time = time.time() - start_time_main
    print(f"Completed in {total_processed_time:.2f} seconds")
    ret_metric_dict['Time'] = total_processed_time

    # Save all metrics (compression ratio, entropy, PSNR, time, etc.) to CSV
    write_dict_to_csv(ret_metric_dict, output_filename=compressed_file_name+f'_{compression_alg}_metrics.csv')

if __name__ == "__main__":
    # Use Fire library to automatically create CLI from main() function
    # Converts function arguments to command-line flags
    fire.Fire(main)
