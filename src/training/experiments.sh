# first round of experiments to fine optimal CNN arch
# python src/training/train.py --model LM
# python src/training/train.py --model 1
# python src/training/train.py --model 10
# python src/training/train.py --model dense
# python src/training/train.py --model PR
python src/training/train.py --model PR-scaled
python src/training/train.py --model PR-super