import streamlit as st
import subprocess
import tempfile
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pytesseract
from PIL import Image
import pdf2image

# Streamlit App
st.set_page_config(page_title="Resume Classifier")
st.header("Resume Analysis")

# Input Text Area
input_text = st.text_area("Enter Job Description: ", key="input")

# File Uploader
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

# Submit Button
submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Run the other script and capture the output
        process = subprocess.Popen(['python', 'app - Copy.py', input_text, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if stderr:
            st.error(f"Error: {stderr}")
        else:
            st.subheader("The Response is")
            st.write(stdout)
    else:
        st.write("Please upload the resume")

def main_logic_code():

    # Load the dataset
    file_path = '5k ish.csv'
    df = pd.read_csv(file_path)

    # Combine relevant columns into a single text column
    df['resume_text'] = df[['Skills', 'Education', 'Years of Experience', 'Experience', 'Courses']].astype(str).agg(' '.join, axis=1)

    # Encode the job titles
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Category'])

    # Split the dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['resume_text'], df['label'], test_size=0.2, random_state=42)

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

    # Create Dataset class
    class ResumeDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]).long()  # Ensure labels are of type Long
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ResumeDataset(train_encodings, train_labels.tolist())
    val_dataset = ResumeDataset(val_encodings, val_labels.tolist())

    # Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Category'].unique()))

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # Updated to the correct argument name
        report_to="none"  # Disable W&B logging
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained('./job_prediction_model')
    tokenizer.save_pretrained('./job_prediction_model')

    # Prediction function
    def predict_resume_probability(resume_text, model, tokenizer, device, label_encoder):
        model.to(device)
        inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
        predicted_probability = probabilities[0][predicted_label_idx].item()
        return predicted_label, predicted_probability, probabilities

    # Function to convert image to text using pytesseract
    def image_to_text(image):
        return pytesseract.image_to_string(image)

    if uploaded_file is not None:
        import io

        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        # Get the text from the PDF
        resume_text = ' '.join([image_to_text(image) for image in images])

        # Make prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_label, predicted_probability, probabilities = predict_resume_probability(resume_text, model, tokenizer, device, label_encoder)

        # Display the results
        st.subheader("The Response is")
        st.write(f"Resume score {predicted_probability * 100:.2f}%")
    else:
        st.write("Please upload the resume")


