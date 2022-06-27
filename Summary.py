print("Importing dependencies...")
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# These are classes

print("Selecting model type...")
model_type = "google/pegasus-xsum"

#Create the 
print("Creating tokenizer...")
tokenizer = PegasusTokenizer.from_pretrained(model_type)

#Load the model
print("Loading the model...")
model = PegasusForConditionalGeneration.from_pretrained(model_type)

#Perform summary

#Input the text
print("Loading text...")
text = """ Self publishing should not have to be so difficult.
Finding an affordable book reviewing and book summary service has actually proved to be almost impossible for self publishing South African Authors.
Detailed book reviews and summaries are important aspects of pre-publishing as they allow you as the author to get a more thorough understanding about your writing techniques, the impact it has, things that could be adjusted to create a more impactful and lasting impression with your readers.
Please email me if you are interested.
The purpose of this service is to give affordable prices to self publishing authors. It's already a pricey and lengthy process to get yourself published. Let me at least help cut the costs in half and make it more affordable to get professional Book Reviews, Book Summaries Ghostwriting and Biographies done.
"""
#Create tokens from the text
print("Creating tokens...")
tokens = tokenizer(text, truncation=True, padding="longest",return_tensors="pt")

#Summarize via tokens
print("Performing summary...")
summary = model.generate(**tokens)

#Decode the summary
print("The summary is: ")
print()
tokenizer.decode(summary[0])