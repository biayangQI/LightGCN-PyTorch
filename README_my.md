1. dataset:
用到了lightgcn下面的train_1_gcn.txt, train_1.txt, eval_1.txt, test_1.txt

2. 运行music 数据集，train_1_gcn.txt, train_1.txt, eval_1.txt, test_1.txt, tpk任务

python main.py --dataset=music --data_index=1 --task=topk

CTR任务\
python main.py --dataset=music --data_index=1 --task=ctr