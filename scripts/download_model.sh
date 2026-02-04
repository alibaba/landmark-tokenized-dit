pip install modelscope
modelscope download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen/Qwen2.5-VL-7B-Instruct
modelscope download Alibaba_Research_Intelligence_Computing/LaTo --local-dir ./models/LaTo

# Or download from huggingface
# pip install huggingface_hub
# export HF_ENDPOINT=https://hf-mirror.com # optional: if you want to use a mirror
# hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen/Qwen2.5-VL-7B-Instruct
# hf download Alibaba-Research-Intelligence-Computing/LaTo --local-dir ./models/LaTo