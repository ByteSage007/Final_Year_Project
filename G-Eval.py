import re

#Defines function for G-Eval benchmark which uses an llm to judge the summaries generated, with a focus on hallucinations 
def g_eval_hallucination(source, summary):
    prompt = f"""
Task:
Evaluate factual consistency using explicit verification.

Steps:
1. Extract up to 5 factual claims from the SUMMARY as questions.
2. Verify each claim against the SOURCE.
3. For each claim, output:
   Verdict: SUPPORTED or UNSUPPORTED

Respond strictly in this format:
Claims:
- Claim 1: <question>
  Verdict: SUPPORTED / UNSUPPORTED
- Claim 2: ...
  Verdict: SUPPORTED / UNSUPPORTED
...

SOURCE:
{source}

SUMMARY:
{summary}
"""
    return mistral_pipe(prompt)[0]["generated_text"]


#Function to calculate G-Eval score
def compute_derived_score(text):
    supported = len(re.findall(r"Verdict:\s*SUPPORTED", text))
    total = len(re.findall(r"Verdict:\s*(SUPPORTED|UNSUPPORTED)", text))
    return supported / total if total > 0 else None


#Main Loop
g_eval_outputs = []
g_eval_scores = []

for i in tqdm(range(len(dataset)), desc="Running G-Eval (derived scoring)"):
    result = g_eval_hallucination(
        dataset[i]["article"],
        generated_summaries[i]
    )

    score = compute_derived_score(result)

    g_eval_outputs.append(result)
    g_eval_scores.append(score)

print("\nSample G-Eval output:\n")
print(g_eval_outputs[0])
print("Derived G-Eval score:", g_eval_scores[0])
