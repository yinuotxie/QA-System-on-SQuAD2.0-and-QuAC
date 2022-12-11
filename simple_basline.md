# OpenDomainQA Simple Baseline
CIS 5300 (Computational Linguistics) Final Project

Our simple baseline retrieves relevant passages with lexical search and generate
the answer with Bert(”bert-base-uncased”) model. As expected, we observed a
poor score with the simple baseline which only reinforces the intuition that more
advanced retriever is needed and the fine-tuning of the Bert model on SQuAD
is necessary.

To execute the simple baseline model, the below command can be used,
python filepath --baseline False

(default argument picks up the strong baseline model)