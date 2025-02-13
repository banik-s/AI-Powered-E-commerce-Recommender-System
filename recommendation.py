import os
import json
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

# Load environment variables
load_dotenv()

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

def load_huggingface_llm():
    return HuggingFaceHub(
        repo_id=HF_MODEL,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
    )

def process_data(refined_df):
    """
    Process the dataset and create a FAISS vector store.
    """
    refined_df['combined_info'] = refined_df.apply(lambda row: f"Product ID: {row['pid']}. Product URL: {row['product_url']}. "
                                                                 f"Product Name: {row['product_name']}. Primary Category: {row['primary_category']}. "
                                                                 f"Retail Price: ${row['retail_price']}. Discounted Price: ${row['discounted_price']}. "
                                                                 f"Primary Image Link: {row['primary_image_link']}. Description: {row['description']}. "
                                                                 f"Brand: {row['brand']}. Gender: {row['gender']}", axis=1)

    loader = DataFrameLoader(refined_df, page_content_column="combined_info")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

def save_vectorstore(vectorstore, directory):
    vectorstore.save_local(directory)

def load_vectorstore(directory, embeddings):
    return FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)

def get_similar_products(vectorstore, query, k=3):
    """
    Retrieve the top-k similar products from FAISS.
    """
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    if not retrieved_docs:
        print("⚠️ No products retrieved from FAISS.")
        return []
    return retrieved_docs

def format_retrieved_products(retrieved_docs):
    """
    Convert FAISS retrieved documents into a structured JSON format.
    """
    product_list = []
    
    for doc in retrieved_docs:
        details = doc.page_content.split(". ")  # Splitting details
        product_info = {
            "Product Name": details[2].replace("Product Name: ", ""),
            "Category": details[3].replace("Primary Category: ", ""),
            "Brand": details[8].replace("Brand: ", ""),
            "Price": float(details[5].replace("Retail Price: $", "")) if "Retail Price: $" in details[5] else "Unknown",
            "Stock Status": "Available" if "Retail Price: $" in details[5] else "Out of Stock"
        }
        product_list.append(product_info)

    return json.dumps(product_list, indent=4)

def extract_cleaned_recommendations(response):
    """
    Extract only valid JSON product recommendations from LLM output.
    """
    try:
        print("🛠 Raw LLM Response:", response)  # Debugging

        json_match = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
        if json_match:
            response = json_match.group(0)  

        products = json.loads(response)

        if not isinstance(products, list) or len(products) == 0:
            return "⚠️ No matching products found. Try a different query."

        formatted_output = "\n\n".join([
            f"🛍️ **Product {i+1}:**\n"
            f"- **Product Name:** {p.get('Product Name', 'N/A')}\n"
            f"- **Category:** {p.get('Category', 'N/A')}\n"
            f"- **Brand:** {p.get('Brand', 'N/A')}\n"
            f"- **Price:** ${p.get('Price', 'N/A')}\n"
            f"- **Stock Status:** {p.get('Stock Status', 'N/A')}\n"
            for i, p in enumerate(products)
        ])

        return formatted_output

    except json.JSONDecodeError as e:
        print("⚠️ JSON Parsing Error:", e)
        return "⚠️ No clean product recommendations found. (Invalid JSON format)"

def display_product_recommendation(refined_df):
    """
    Display product recommendation section.
    """
    st.header("Product Recommendation")

    vectorstore_dir = 'vectorstore'

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(vectorstore_dir):
        vectorstore = load_vectorstore(vectorstore_dir, embeddings)
    else:
        vectorstore = process_data(refined_df)
        save_vectorstore(vectorstore, vectorstore_dir)

    llm = load_huggingface_llm()

    department = st.text_input("Product Department")
    category = st.text_input("Product Category")
    brand = st.text_input("Product Brand")
    price = st.text_input("Maximum Price Range")

    if st.button("Get Recommendations"):
        query = f"Department: {department}, Category: {category}, Brand: {brand}, Price: {price}"
        retrieved_products = get_similar_products(vectorstore, query)

        if not retrieved_products:
            st.write("⚠️ No matching products found. Try a different query.")
            return

        retrieved_json = format_retrieved_products(retrieved_products)

        response = llm.invoke(f"""
        You are an AI-powered product recommendation assistant.

        Here are the retrieved product listings:

        {retrieved_json}

        🚨 IMPORTANT: Return **only valid JSON** in this format:

        [
            {{
                "Product Name": "<Retrieved Product Name>",
                "Category": "<Retrieved Product Category>",
                "Brand": "<Retrieved Product Brand>",
                "Price": <Retrieved Product Price>,
                "Stock Status": "Available" or "Out of Stock"
            }},
            ...
        ]

        ⚠️ STRICT RULES:
        - **Only output valid JSON.** No explanations, markdown, or extra text.
        - **Use retrieved listings only.** Do NOT modify product details.
        - **If no matching products exist, return an empty JSON array `[]`.**
        - **Do NOT include any words before or after the JSON output.**
        """)

        formatted_response = extract_cleaned_recommendations(response)
        st.write(formatted_response)
