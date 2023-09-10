nvidia-smi

cd ../../evaluation/samsum

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('ENTER YOUR HF ID TOKEN')"

python3 eval_samsum_4bit_bnb.py --model_name huggyllama/llama-7b --adapter ENTER YOUR ADAPTER --seed 42 --file_name llama_sam_4bit_7b_bnb_seed42.txt
