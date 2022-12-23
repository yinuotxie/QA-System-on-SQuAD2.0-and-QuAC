"""
Document Reader File.
"""
import numpy as np
import torch


class QuestionAnswering(object):
    def __init__(self, tokenizer, reader, retriever=None, baseline=False):
        """
        @param: tokenizer: tokenizer to use in the model
        @param: retriever: retriever to use in the model, if apply_retriever is True
        @param: reader: reader to use in the model
        @param: baseline: bool, whether to use the baseline model (randomly generate the answer)
        @param: apply_retriever: bool, whether to use retriever to select contexts instead of
                                       using the give context
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.reader = reader.to(self.device)
        self.baseline = baseline

    def question_answer(self, question, context=None, display=True, top_k=5):
        """
        Generate an answer to the question
        @param: question: str, user's question
        @param: context: str, question's context if provided, default None
        @param: display: bool, whether to display the search results
        @param: top_k: int, if using retriever, the number of contexts to select
        """
        self.reader.eval()

        with torch.no_grad():
            if self.retriever is None and context is None:
                raise Exception("Either retriever or context must be provided to generate an answer!")

            if self.baseline or self.retriever is None:
                inputs = self.tokenizer(question, context, truncation='only_second',
                                        return_tensors="pt", max_length=512)
                input_ids = inputs["input_ids"].tolist()[0]

                if self.baseline:
                    # applying baselin, randomly select start and end pos
                    answer_start, answer_end = np.random.choice(np.arange(len(input_ids)), 2)
                else:
                    outputs = self.reader(**inputs.to(self.device))
                    answer_start = torch.argmax(outputs[0])
                    answer_end = torch.argmax(outputs[1]) + 1
            else:
                # applying retriever
                topk_context = self.retriever.search(question, top_k=top_k)

                if display:
                    print("========================================\n")
                    print("Search Results")
                    for i, context in enumerate(topk_context):
                        print(f"Top {i}: {context}")
                    print("========================================\n")

                inputs = self.tokenizer(question, topk_context[0], truncation='only_second',
                                        return_tensors="pt", max_length=512)
                outputs = self.reader(**inputs.to(self.device))
                best_prob = torch.max(outputs[0]).item() * torch.max(outputs[1]).item()
                answer_start = torch.argmax(outputs[0])
                answer_end = torch.argmax(outputs[1]) + 1
                input_ids = inputs["input_ids"].tolist()[0]

                for context in topk_context[1:]:
                    inputs = self.tokenizer(question, context, truncation='only_second',
                                            return_tensors="pt", max_length=512)
                    outputs = self.reader(**inputs.to(self.device))

                    prob = torch.max(outputs[0]).item() * torch.max(outputs[1]).item()
                    # update to the better answer
                    if best_prob < prob:
                        best_prob = prob
                        answer_start = torch.argmax(outputs[0])
                        answer_end = torch.argmax(outputs[1]) + 1
                        input_ids = inputs["input_ids"].tolist()[0]

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start: answer_end]))

            return answer
