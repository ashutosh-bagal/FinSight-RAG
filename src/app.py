import streamlit as st
import requests

st.set_page_config(page_title="VaultRAG — Finance Filings Assistant", page_icon="📊")

st.title("FinSightRAG — Finance Filings Assistant")
st.caption("Ask questions about Apple and Amazon's 10-K filings")

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "answer" not in st.session_state:
    st.session_state.answer = None

role = st.selectbox("Select your role", ["public", "finance"])
question = st.text_input("Ask a question")

ask_clicked = st.button("Ask", disabled=st.session_state.is_loading)

if ask_clicked:
    if question:
        st.session_state.is_loading = True
        st.session_state.answer = None
        st.rerun()
    else:
        st.warning("Please enter a question")

if st.session_state.is_loading:
    with st.spinner("Checking access and searching filings..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask", json={"question": question, "role": role}
            )
            st.session_state.answer = response.json()["answer"]
        except Exception as e:
            st.session_state.answer = f"Error: {e}"
    st.session_state.is_loading = False
    st.rerun()

if st.session_state.answer:
    st.write(st.session_state.answer)
