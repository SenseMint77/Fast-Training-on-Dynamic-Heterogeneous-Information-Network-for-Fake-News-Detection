# Fast_Training_on_Dynamic_Heterogeneous_Information_Network_for_Fake_News_Detection

1. Install packages from root with conda. 

For `osx-arm64`:
```
conda create --name gnn_fnd --file gnn_fnd_osx-arm64.txt
conda activate gnn_fnd
```

For `linux64`:
```
conda create --name gnn_fnd --file gnn_fnd_linux64.txt
conda activate gnn_fnd
```

or make sure you have the following packages installed:
  * NumPy
  * SciPy
  * PyTorch
  * PyTorch Geometric
  * Huggingface Transformers
  * Scikit-learn
  * Matplotlib
  * Gensim (For experiments with other models using Word2Vec)

2. Extract and preprocess data set

Set `data_set` to `liar_dataset` or `FakeNewsNet`((https://github.com/KaiDMML/FakeNewsNet).
If you would like to fine tune BERT, set `with_bert_finetuning = True` in `extract_data.py`. 

```
cd data/script
python extract_data.py
```

3. Change parameters in `main.py`:

  `with_author_test = True` to include author information during test.
  `data_set` to `liar_dataset` or `FakeNewsNet`.
  `version` to `no_finetuning` or `with_finetuning`.
  `model_type` to `bipartite` or `heterogeneous`. 

4. Run model from root
```
python main.py
```

Once complete, a plot of training loss in the K-fold cross validation is generated in `loss.png`. Example:
![](https://github.com/v-raina/GNN_Fake_News_Detection/blob/main/loss.png?raw=true)

