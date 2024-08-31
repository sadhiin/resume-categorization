# Resume Categorization Project

## Project Overview

This project aims to automatically categorize resumes based on their domain (e.g., sales, marketing) using machine learning techniques. The system processes a batch of resumes, categorizes them, and outputs results to both directory structures and a CSV file.

## Model Selection

For this project, we chose to use a ML [Random Forest Classifier] due to its ability to handle high-dimensional data, good performance on classification tasks.

## Preprocessing and Feature Extraction

The preprocessing pipeline includes the following steps:

1. Text Cleaning: Remove special characters, numbers, extra whitespace, links, emails so on.
2. Tokenization: Split the text into individual words or tokens.
3. Lowercasing: Convert all text to lowercase to ensure consistency.
4. Stop Word Removal: Remove common words that don't contribute much to the classification.
5. Stemming: Reduce words to their root form to capture similar words.

For feature extraction, we use `TF-IDF` (Term Frequency-Inverse Document Frequency) vectorization. This method captures the importance of words in each resume relative to the entire corpus, which is particularly useful for document classification tasks.

## Running the Script

To run the script, follow these steps:

1. Ensure you have Python 3.11 installed on your system.
2. Install the required dependencies:
`pip install -r requirements.txt`

3. Place the resumes you want to categorize in a directory.
4. Run the script from the command line:
python script.py path/to/resume/directory


## Expected Outputs

After running the script, you can expect the following outputs:

1. The resumes will be moved into subdirectories based on their categorized domain within the input directory.
2. A CSV file named `categorized_resumes.csv` will be created in the input directory, containing two columns: `filename` and `category`.

## Evaluation Metrics

The model's performance was evaluated using the following metrics:

- Accuracy: `66.0`%
- Precision: `63.0`%
- Recall: `66.0`%
- F1-score: `63.0`%


## Future Improvements

- Advanced NLP techniques like word embeddings or transformers for better feature representation.
- Explore ensemble methods to potentially improve classification accuracy.
