

python main_unet.py \
     --mode test \
     --weight_path Unet/ \
     --test_noisy ../spec/test/noisy_03.txt \
     --test_clean ../spec/test/clean.txt \
     --result_path 03 \
     --gpus cuda:0 \

python main_unet.py \
     --mode test \
     --weight_path Unet/ \
     --test_noisy ../spec/test/noisy_04.txt \
     --test_clean ../spec/test/clean.txt \
     --result_path 04 \
     --gpus cuda:0 \



python main_unet.py \
     --mode test \
     --weight_path Unet/ \
     --test_noisy ../spec/test/noisy_06.txt \
     --test_clean ../spec/test/clean.txt \
     --result_path 06 \
     --gpus cuda:0 \

python main_unet.py \
     --mode test \
     --weight_path Unet/ \
     --test_noisy ../spec/test/noisy_07.txt \
     --test_clean ../spec/test/clean.txt \
     --result_path 07 \
     --gpus cuda:0 \


python main_unet.py \
     --mode test \
     --weight_path Unet/ \
     --test_noisy ../spec/test/noisy_09.txt \
     --test_clean ../spec/test/clean.txt \
     --result_path 09 \
     --gpus cuda:0 \

python main_unet.py \
     --mode test \
     --weight_path Unet/ \
     --test_noisy ../spec/test/noisy_10.txt \
     --test_clean ../spec/test/clean.txt \
     --result_path 10 \
     --gpus cuda:0 \




