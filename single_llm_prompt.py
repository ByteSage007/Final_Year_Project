#We are loading a model and using it to generate summaries of the documents in a dataset
#In this example we are using BART


#Loading the dataset

bart_model_name = "facebook/bart-large-cnn"

bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    bart_model_name
)

#Function for generating summaries

def generate_summary(article):
    prompt = (
        "Provide a concise summary of the text in around 150 words. "
        "Output the summary text and nothing else.\n\n"
        + article
    )

    inputs = bart_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(bart_model.device)

    summary_ids = bart_model.generate(
        **inputs,
        max_length=200,
        min_length=120,
        num_beams=4,
        early_stopping=True
    )

    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

generated_summaries = []

#Function call

for item in tqdm(dataset, desc="Generating summaries"):
    summary = generate_summary(item["article"])
    generated_summaries.append(summary)

#Printing the first summary

print("\nSample summary:\n")
print(generated_summaries[0])