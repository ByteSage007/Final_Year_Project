
#Function definition for a light version of QAGS. It prompts an llm to use the summaries to generate queries and answer then in order to measure hallucination 
def llm_qa_probe(source, summary):
    prompt = f"""
Task:
Evaluate factual consistency using question answering.

Steps:
1. Generate 5 factual questions from the SUMMARY.
2. Answer each question using ONLY the SOURCE text.
3. For each answer, output:
   Verdict: SUPPORTED or UNSUPPORTED

Respond strictly in this format:
Questions:
- Q1: ...
  A1: ...
  Verdict: SUPPORTED / UNSUPPORTED
- Q2: ...
  A2: ...
  Verdict: SUPPORTED / UNSUPPORTED
...

SOURCE:
{source}

SUMMARY:
{summary}
"""
    return mistral_pipe(prompt)[0]["generated_text"]


#Main Loop
qa_probe_outputs = []
qa_probe_scores = []

for i in tqdm(range(len(dataset)), desc="Running LLM-QA probing (derived scoring)"):
    result = llm_qa_probe(
        dataset[i]["article"],
        generated_summaries[i]
    )

    score = compute_derived_score(result)

    qa_probe_outputs.append(result)
    qa_probe_scores.append(score)

print("\nSample LLM-QA output:\n")
print(qa_probe_outputs[0])
print("Derived QA score:", qa_probe_scores[0])
