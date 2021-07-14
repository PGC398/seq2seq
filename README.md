# seq2seq
```
##数据加载部分 - dataLoader.py
```
**example**
```
'预郜杀手', '预告杀手'
'将京尤着看吧', '将就着看吧'
'因为队长太shuai了 所以是一𫝀颗星', '因为队长太帅了 所以是五颗星'
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
