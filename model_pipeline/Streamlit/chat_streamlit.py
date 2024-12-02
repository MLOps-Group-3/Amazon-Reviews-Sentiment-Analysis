import streamlit as st
from llm_fetch import process_text_query

# Streamlit app configuration
st.set_page_config(
    page_title="Amazon Reviews Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar filters
st.sidebar.header("Filters")
category = st.sidebar.text_input("Category", "")
subcategory = st.sidebar.text_input("Subcategory", "")
year = st.sidebar.text_input("Year", "")
month = st.sidebar.text_input("Month", "")  

# Main page header
st.title("Amazon Reviews Chatbot")
st.markdown(
    "This chatbot analyzes sentiment and provides insights based on product reviews. "
    "Use the filters in the sidebar for a more refined search."
)

# User input for chatbot query
user_input = st.text_input("Ask me a question about Amazon reviews:", "")

# Set top_k for results
top_k = st.sidebar.slider("Number of Results", min_value=1, max_value=20, value=10)

# Chatbot response
if st.button("Get Response"):
    if user_input.strip():
        with st.spinner("Processing your query..."):
            response = process_text_query(
                input_text=user_input,
                top_k=top_k,
                category=category,
                subcategory=subcategory,
                year=year,
                month=month,
            )
        st.subheader("Chatbot Response")
        st.write(response)
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown(
    "Powered by **OpenAI**, **Pinecone**, and **Google Cloud Platform**. "
    "Built with 💡 by your data team."
)
