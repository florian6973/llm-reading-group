
# print(model)

import re

# https://huggingface.co/docs/datasets/en/create_dataset
# https://huggingface.co/docs/datasets/tabular_load#csv-files

def remove_references(text):
    # Remove text after "REFERENCES"
    reference_start = text.find("REFERENCES")
    if reference_start != -1:
        text = text[:reference_start]
    
    return text

def remove_links(text):
    pattern = r"\[\d+\]|\(http[s]?://\S+\)|www\.\S+|[^a-zA-Z0-9\s]" 
    return re.sub(pattern, "", text)

def remove_special_chars(text):
    pattern = r"[^\w\s.]"  
    return re.sub(pattern, "", text)

def preprocess_text(textso):
    texts = []
    for text in textso:
        text = remove_references(text)
        text = remove_links(text)
        text = text.lower() 
        text = remove_special_chars(text)

        text = re.sub(r'\[\d*\]', '', text)  # Remove square brackets containing numbers
        text = re.sub(r'\[.*?\]', '', text)   # Remove other text between square brackets
        
        # Remove occurrences of "fig"
        # text = re.sub(r'\bfig.\b', '', text)
        
        
        # # Remove numbers
        # text = re.sub(r'\b\d+\b', '', text)  # Remove numbers

        # # Remove single characters or numbers in a line
        # text = re.sub(r'\b\w\b|\b\d\b', '', text)

        # Filter out lines with only a single character, number, or special character
        # lines = text.split('\n')
        # lines = [line for line in lines if len(line.strip()) > 1]  # Filter out lines with length <= 1
        # text = '\n'.join(lines)
        

        texts.append(text)

    return texts


def get_preprocess_function(tokenizer):
    def preprocess_function(examples):
        text = examples['text']
        return tokenizer(
            ["### Instruction: generate a clinical note.\n\n### Answer:\n" + t + tokenizer.eos_token for t in text], #preprocess_text(text)],
            return_tensors='pt',
            max_length=4096, #4096,  # Adjust the max length as needed
            truncation=True, padding="max_length",
        )
    # how many tokens is a note
    return preprocess_function
