import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()
# 1. Load the Elfie.co data
loader = TextLoader('data/elfieco.txt')
docs = loader.load()

# 2. Split Texts into Embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)
texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# 3. Funtion for similarity search
def retrieve_info(query):
    similar_responses = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_responses]
    return page_contents_array

# 4. Setup LLMChain, Prompts, Template
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

template = """Question: {question}

Answer: Let's think step by step. {rag_text}"""

prompt = PromptTemplate(template=template, input_variables=["question","rag_text"])
chain = LLMChain(llm=llm, prompt=prompt)

# 5. RAG
def generate_response(query):
    similar_responses = retrieve_info(query)
    response = chain.run(question=query, rag_text=similar_responses)
    return response

# 5. Build a chatbot
def main():
    st.set_page_config(
        page_title="Elfie.co Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="auto"
    )
    
    st.header("Ask Anything About Elfie.co")
    message = st.text_area("Example: 'Who is Elfie.co founders?'")
    
    if message:
        st.write("Answer:")
        result = generate_response(message)
        st.info(result)
        
if __name__ == "__main__":
    main()