# OpenDomainQA Strong Baseline
CIS 5300 (Computational Linguistics) Final Project

In an attempt to make use of relevance in the extracted paragraphs, we fetched
the top 5 ranking paragraphs from the comprehension. We encode all passages
into our vector space and then perform semantic search where the queries are
encoded using the bi-encoder to find potentially relevant passages.

The input query and passage pair are tokenized with WordPiece embeddings,
and then packed into a sequence. Then feed the final input to BERT will be the
sum of text embeddings, segment embeddings and position embeddings. The
input then goes through layers of bidirectional Transformers to produce hidden
states that can be used as contextual representation of input query passage
pair. From the available pre-trained BERT models, we have used BERT-Large, Uncased (Whole Word Masking) with 24-layer, 1024-hidden, 16-heads and 340M
parameters, for this task.

The main function executes the model with the strong baseline implementation by default.
