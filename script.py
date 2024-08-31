import os
import sys
import re
import argparse
import pandas as pd
import pickle
import nltk
import shutil
from PyPDF2 import PdfReader
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

saved_models_dir = 'models'

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return text

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)

    # non-alphanumeric characters, punctuation, digits, and extra whitespace
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'[^\w\s]|_', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+\s', "", text)
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "", text)

    text = text.lower()
    words = word_tokenize(text)

    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)


def categorize_resumes(resume_dir,output_directory):
    categorized_resumes = {}
    # loading the saved pkls for label encoder, tfidf vectorizer and the model
    with open(os.path.join(saved_models_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)

    with open(os.path.join(saved_models_dir,"tfidf_vectorizer.pkl"), 'rb') as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(saved_models_dir,"RandomForestClassifier_best_model.pkl"), 'rb') as f:
        loaded_model = pickle.load(f)

    for filename in os.listdir(resume_dir):

        if filename.endswith(('.pdf')):
            resume_text = extract_text_from_pdf(os.path.join(resume_dir, filename))
            resume_text = preprocess_text(resume_text)
            resume_vector = vectorizer.transform([resume_text])

            predicted_label = loaded_model.predict(resume_vector)[0]
            predicted_category = label_encoder.inverse_transform([predicted_label])[0]
            categorized_resumes[filename] = predicted_category

    df = pd.DataFrame(list(categorized_resumes.items()), columns=['filename', 'category'])
    df.to_csv('categorized_resumes.csv', index=False)
    print("Categorized resumes saved to 'categorized_resumes.csv'")

    print(df)

    # resumes to their respective category folders
    if output_directory is None:
        output_directory = 'categorized_resumes'

    for filename, category in categorized_resumes.items():
        category_dir = os.path.join(output_directory, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            # copy the file to the category directory
        shutil.copyfile(os.path.join(resume_dir, filename), os.path.join(category_dir, filename))



def main():
    parser = argparse.ArgumentParser(description="Categorize resumes based on their content using a trained model.")
    parser.add_argument("input_directory", help="Path to the directory containing resumes to be categorized.")
    parser.add_argument("-o", "--output_directory", default="categorized_resumes",
                        help="Path to the directory where categorized resumes will be saved. Default is 'categorized_resumes'.")

    args = parser.parse_args()

    if not os.path.exists(args.input_directory):
        print(f"Error: The input directory '{args.input_directory}' does not exist.")
        sys.exit(1)

    categorize_resumes(args.input_directory, args.output_directory)

if __name__ == '__main__':
    main()