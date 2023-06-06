CUDA_VISIBLE_DEVICES=4,5,6,7 python inference.py \
    --model_name_or_path ../mozi-7b-3m-40k \
    --foundation_model bloom \
    --test_file IPQA-test-5.json \
    --predictions_file ./mozi-7b--predictions-ipqa-5.json


   # /data6/.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348
   # /data6/model/BELLE-7B-2M
   # CUDA_VISIBLE_DEVICES=7
   # /data6/.cache/huggingface/hub/models--bigscience--bloomz-7b1-mt/snapshots/13e9b1a39fe86c8024fe15667d063aa8a3e32460
   # /data6/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/35ca52301fbedee885b0838da5d15b7b47faa37c
   # /data6/mozi-7b-3m-40k
