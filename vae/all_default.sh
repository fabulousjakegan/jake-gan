# bash download.sh
# pip install -U pip
# pip install -r requirements.txt
python3 analyzer.py
python3 build.py
python3 main.py \
  --model ConvVAE \
  --trainer VAETrainer \
  --architecture architecture-vae.json
# python convert.py \
#   --src SF1 \
#   --trg TM3 \
#   --model ConvVAE \
#   --checkpoint logdir/train/[timestamp]/[model.ckpt-[id]] \
#   --file_pattern "./dataset/vcc2016/bin/Testing Set/{}/*.bin"
# echo "Please find your results in `./logdir/output/[timestamp]`"
