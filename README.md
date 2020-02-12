# summarization-humanitarian-documents
A graph-based approach and a neural-based approach for humanitarian documents summarization. 

[Text Rank](https://www.aclweb.org/anthology/W04-3252.pdf) is a graph based model, that using text units as vertices and the similarity between the text units as the edges, rank the text units, from most relevant to least relevant. Given this ranking, and a fixed length of the desired output summary, the model produces a summary in an unsupervised manner. 

[NeuSum](https://www.aclweb.org/anthology/P18-1061.pdf) is a neural based model, that employs a hierarchical document encoder, to get sentences representations, and a sentence extractor, that from the sentences representations, goes extracting, step by step a new sentence, according to its overall contribution to the performance of the current summary in a supervised manner.

For the word representation the performance of out domain and in domain [GloVe embeddings](https://nlp.stanford.edu/pubs/glove.pdf) are compared.

The data used for such comparison belongs to the [DEEP platform](https://deephelp.zendesk.com/hc/en-us/articles/360015943731-What-is-DEEP-).
