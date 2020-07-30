# Evolutionary Algorithms
- Original paper: [Using evolutionary algorithms as instance selection for data reduction in KDD: an experimental study](https://ieeexplore.ieee.org/document/1255391)
- This work implement CHC algorithm

# Requirements
- Python 3.6

# Training and Testing
Training 
```bash
python main.py
```

Testing
```bash
python evalution.py 
```

# Result
| source | % reduction  | ACC  |
| :-----: | :-: | :-: |
| Paper | ~99.29 | 94.18 -> 93.53 |
| Our | 54.26 | 97.74 -> 97.17 |
