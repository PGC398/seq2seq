# seq2seq
```
##数据加载部分 - dataLoader.py
```
```
##模型定义部分 - seq2seq.py
```

```
##模型训练部分 - train.py
```
**使用**
```
python train.py --epoch_num 1 --embedding_size 300 --hidden_size 300
```

```
##模型测试部分 - evalute.py
```
**使用**
```
python evaluate.py --encoder model/encoder.pth --decoder model/decoder.pth
