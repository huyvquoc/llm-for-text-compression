# Course project: use LLM for compression

My modifications on the LLMZip repo for running experiments for my course project. Please have a look at the authors' original repo [LLMZip](https://github.com/vcskaushik/LLMzip) for more information.

Please read the original paper [LLMZip: Lossless Text Compression using Large Language Models](https://arxiv.org/abs/2306.04050)

 
## Setup

This repository is identical to the [LLaMA repository] (https://github.com/facebookresearch/llama) with additional scripts to perform compression. The setup is identical to that of LLaMA. LLaMA Setup is included below for ease of access

## Compression

The code below can be used for compressing any text file ($TEXT_FILE) using LLaMa and Arithmetic Coding , the resulting compressed file will be stored in a specified folder ($COMPRESSION_FOLDER). $TARGET_FOLDER is the folder with LLaMa weights and tokenizer.

* Compression and Decompression
  
```
torchrun --nproc_per_node 1 LLMzip_run.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model --win_len 511 --text_file $TEXT_FILE --compression_folder $COMPRESSION_FOLDER 

```

* Compression Only

```
torchrun --nproc_per_node 1 LLMzip_run.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model --win_len 511 --text_file $TEXT_FILE --compression_folder $COMPRESSION_FOLDER --encode_decode 0

```
* Additional Flags (**Default***)
  * compression_alg -  **ArithmeticCoding*** / RankZip / both
  
  * encode_decode - 0: Only encode, 1: only decode, **2: both***
  
  * batched_encode - True, **False***  |  !! Use only for faster encoding (theoretical entropy computations), as decoding doesn't work with batched encoding. !!
  
  * with_context_start - True, **False*** | avoids encoding the initial context and provides the initial context at the decoder
  
  * verify_save_decoded - 0: don't verify/save, 1: only verify, **2: verify and save***
  

### Arithmetic Coding
The arithmetic coding implementation is from [Deep Zip](https://github.com/mohit1997/DeepZip) repo , which is based of the implementation by [Project Nayuki](https://github.com/nayuki/Reference-arithmetic-coding)

# Llama Setup

In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

```sh
conda create -n llama_zip python=3.8
```

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## My comments:
Please have a look at the `scripts` subfolder for commands to run this project:

```sh
./scripts/run1.sh
./scripts/run2.sh
./scripts/run1_only_binary.sh
./scripts/run2_only_binary.sh
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
