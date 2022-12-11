# Open Domain Question Answering System
## Introduction
<p align="justify">
Our project revolves around open domain question answering where we begin with baselines and then explore pre-trained 
models that find answers to a given question. The system is composed of two parts: Document Retriever and Document Reader. 
Document retriever will retrieve the top k relevant passages to the given question. Document reader will then generate 
an answer based on these passages. The answers to the questions are derived as a subset of the comprehension. We are using 
the Stanford Question Answering Dataset(SQuAD2.0) which is one of the most worked upon large-scale, labeled datasets for 
the project. Here, we have started with a simple retriever baseline where we use lexical search to find the relevant 
passages and a simple reader where we use a Bert model that is trained on SQuAD2.0 dataset.
</p>