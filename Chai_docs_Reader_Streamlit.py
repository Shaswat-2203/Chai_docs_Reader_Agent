import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import google.generativeai as genai

st.set_page_config(page_title="AI Chai docs", layout="wide")

st.title("Chat with Chai docs")

# Sidebar for settings
st.sidebar.header("Settings")

api_key = st.sidebar.text_input("Your Gemini API Key", type="password")
collection_name = "chai_docs"


embedding_model = None
vector_store = None
vector_db = None
chat_model = None

if api_key:
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    if True:
        vector_db = QdrantVectorStore.from_existing_collection(
            url="https://e59dfb81-bb98-4eb6-9806-f172c977a89f.us-east-1-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.EhU0hmKfZ9p-LYubvLHcF7aQg-piIYam2L1qK7CdAnE",
            collection_name="chai_docs",
            embedding=embedding_model
        )

    if vector_db:

        genai.configure(api_key=api_key)

        user_input = st.chat_input("Ask a question")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["parts"][0])

        if user_input:

            with st.chat_message("user"):
                st.markdown(user_input)

            search_results = vector_db.similarity_search(query=user_input)

            context = "\n\n\n".join(
                [f"Page Content: {result.page_content}\nWebsite Link: {result.metadata['source']}" for result in
                 search_results])

            SYSTEM_PROMPT = f"""
                You are a helpfull AI Assistant who answers user query based on the available context
                retrieved from multiple websites along with page_contents and website link.

                You should only ans the user based on the following context and navigate the user
                to open the right web page to know more.

                Context:
                {context}
            """
            chat_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={"temperature": 0},
                system_instruction=SYSTEM_PROMPT
            )

            chat = chat_model.start_chat(history=st.session_state.chat_history)

            # print(SYSTEM_PROMPT)

            response = chat.send_message(user_input)

            with st.chat_message("assistant"):
                st.markdown(response.text)

            # print(response.text)

            st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
            st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
else:
    st.info("Please enter your Gemini API key to begin.")
