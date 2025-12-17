from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# Define the specific Hugging Face model checkpoint for FactCC
factcc_model_name = "manueldeprada/FactCC"

# Load the tokenizer associated with the FactCC model
factcc_tokenizer = AutoTokenizer.from_pretrained(factcc_model_name)

# Load the sequence classification model and automatically assign it to the available device (GPU/CPU)
factcc_model = AutoModelForSequenceClassification.from_pretrained(
    factcc_model_name,
    device_map="auto"
)

# Create a text classification pipeline to handle preprocessing, inference, and truncation
factcc_pipe = pipeline(
    "text-classification",
    model=factcc_model,
    tokenizer=factcc_tokenizer,
    truncation=True
)

# Initialize an empty list to store the evaluation results
factcc_scores = []

# Iterate through the dataset using tqdm to display a progress bar
for i in tqdm(range(len(dataset)), desc="Running FactCC"):
   
    input_text = dataset[i]["article"] + " [SEP] " + generated_summaries[i]
    result = factcc_pipe(input_text)[0]
    factcc_scores.append(result)

# Print a header for the output verification
print("\nSample FactCC output:\n")

# Print the first result to verify the structure of the output
print(factcc_scores[0])
