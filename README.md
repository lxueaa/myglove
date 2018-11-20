## Requirements

Python 3.6

PyTorch 0.4.1 

NLTK

Matplotlib



## Running the Project

The GLoVe model can be trained with:

```python
python demo.py
```


## To Do List

### Recently

- [x] analogy
- [x] similarity
- [ ] fit glove to our model
- [ ] named entity recognition
- [ ] part of speech tagging


## Preliminary Results

#### 2018/11/04: Glove

- dataset: refine_part (part of enwik9 dataset)

- tokens: 15,789,556 (15M)

- vocabs:  207,754 (200K)

- nonzeros:  53,415,770

- epoch:  25

- batch:  104,328  | batch_size:  512

- iter:  2,608,200
- embedding dimension: 32
- window size: 10
- x_max = 100
- alpha = 0.75
- lr = 0.001
- **WordSim353:** 0.4590 (spearman rank) good
- **RW:** 0.2176 (spearman rank) good
- **Word Analogy:** 0.0661 (accuracy) need check

[glove-lr0.001-refine_part.result](data/glove-lr0.001-refine_part.result) (detailed)

#### 2018/11/05: Glove

- dataset: enwik9 dataset
- tokens: 80,237,578 (80M) |  (6B)
- vocabs: 101,654 (100K) | (400k)
- nonzeros:  134,450,281 
- epoch:  11 | (50)
- batch:  262,599  | batch_size:  512
- embedding dimension: 100 | (300)
- window size: 10
- x_max = 100
- alpha = 0.75
- lr = 0.0005 | 0.005
- **WordSim353:** 0.5745 (spearman rank) good  | (65.8)
- **RW:** 0.3035 (spearman rank) good | (38.1)
- **Word Analogy:** 0.0856 (accuracy) need check 

[glove-lr0.0005-refine_enwik9_10M.result](data/glove-lr0.0005-refine_enwik9_10M.result)(detailed)
