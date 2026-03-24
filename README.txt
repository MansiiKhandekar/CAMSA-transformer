Context-Adaptive Multi-Stride Attention (CAMSA)

A custom Transformer-based architecture that introduces multi-stride structured attention to improve interpretability and efficiency over standard self-attention.

🧠Inspiration

Transformers revolutionized NLP after the release of Attention Is All You Need. However, standard self-attention is:

Computationally expensive
Fully dense (attends to all tokens equally)
Often difficult to interpret

This project explores a structured alternative.

💡 Key Idea

Instead of attending to all tokens uniformly, CAMSA introduces:

Multi-Stride Attention

The model attends to tokens at varying strides:

Local context → small strides
Global context → larger strides

This creates structured sparsity in attention.

⚙️ Architecture
Token Embedding
Positional Encoding
Transformer Blocks
Custom CAMSA Attention
Classification Head
- Experimental Setup
Dataset: IMDB Movie Reviews
Task: Sentiment Classification
Framework: PyTorch
- Results
Model	Test Accuracy
Standard Transformer	83.4%
CAMSA Transformer	84.0%+

- Attention Visualization
CAMSA Attention
Standard Attention

Key Insight

CAMSA produces structured and selective attention patterns, while standard attention is more diffuse and less interpretable.

- Features
Custom Transformer implementation
Novel attention mechanism
Attention visualization
Baseline comparison
- Setup
git clone <your-repo-link>
cd CAMSA-Transformer
pip install -r requirements.txt
- Training
python -m training.train (pay attention to the USE_CAMSA in the config.py file for using the CAMSA/Standard model)
- Visualization
python visualize.py
- Future Work
Learnable stride selection
Extension to larger datasets (e.g., GLUE)
Integration with pretrained models
- Author
Built from scratch as part of an exploration into improving Transformer attention mechanisms.
-Citations
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

⭐ If you like this project, consider starring the repo!