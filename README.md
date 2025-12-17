This repository contains the functions for generating llm summaries using two techniques. 

The first technique involves generating summaries using one llm at a time.

The second technique involves the following steps:
(a) generating multiple summaries using multiple llms.
(b) prompting a third llm to use the summaries generated in the first step and the source document to create a better summary focused on reducing hallucinations.

Our results show a measurable improvemnt when multi llms are used.

The evaluation benchmarks used are 
(a) Factcc
(b) G-eval 
(c) LLM-QA
