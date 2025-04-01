import streamlit as st
import pdfplumber
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

from dotenv import load_dotenv
load_dotenv()

def extract_pdf_content(pdf_path):
    """Extract text content from a PDF file."""
    extracted_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n\n"
    
    return extracted_text

def refine_with_groq(extracted_text, api_key):
    """Process PDF text using LangChain's ChatGroq integration."""
    # Initialize the ChatGroq model
    llm = ChatGroq(
        model_name="llama3-70b-8192",  # Using Llama 3 70B model via Groq
        temperature=0.1,  # Low temperature for more deterministic output
        max_tokens=4000,
        api_key=api_key
    )
    
    # Prepare the prompt
    prompt = (
        "You are given extracted text from a PDF. Convert this to properly formatted Markdown. "
        "Please follow these specific guidelines:\n"
        "1. Format all tables as Markdown tables\n"
        "2. Make all headings bold using Markdown syntax (e.g., **Heading**). After the heading , the description must start from next line.\n"
        "3. Preserve the sequential order of all content\n"
        "4. Do not modify or summarize any content - keep all information intact\n"
        "5. Only improve formatting, not content\n\n"
        "Here is the extracted text:\n\n" + extracted_text
    )
    
    try:
        # Send the request to Groq via LangChain
        response = llm([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error processing PDF with Groq API: {str(e)}"

# Set page title and description
st.title("📄 PDF to Markdown Converter")
st.subheader("Powered by Groq API (LangChain Integration)")

# API key handling
api_key_input = "Groq_API_KEY"
# If you want to use a hardcoded key for testing, uncomment and replace below
# api_key = "your_api_key_here"  # Replace with your actual API key
# api_key_input = st.text_input("Enter Groq API Key", value=api_key, type="password")

# File upload widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Process the PDF if a file is uploaded and an API key is provided
if uploaded_file is not None and api_key_input:
    with st.spinner("Processing PDF - this may take a few moments..."):
        # Extract text from the PDF
        extracted_text = extract_pdf_content(uploaded_file)
        
        # Process the text with Groq
        formatted_markdown = refine_with_groq(extracted_text, api_key_input)
        
        # Display the results
        st.subheader("📝 Formatted Markdown Output")
        st.text_area("Markdown", formatted_markdown, height=400)
        
        # Provide download option
        st.download_button(
            label="📥 Download Markdown",
            data=formatted_markdown,
            file_name=f"{uploaded_file.name.split('.')[0]}.md",
            mime="text/markdown"
        )
elif uploaded_file is not None and not api_key_input:
    st.warning("Please enter your Groq API key to process the PDF.")