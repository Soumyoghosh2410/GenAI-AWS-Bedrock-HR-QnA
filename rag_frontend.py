import streamlit as st
import rag_backend as demo
st.set_page_config(page_title="HR QnA ith RAG")

new_title = '<p style = "font-family:sans-serif; color : Green; font-size : 42px;">HR Q & A with RAG </p> '
st.markdown(new_title, unsafe_allow_html=True)

if 'vector_index' not in st.session_state:
    with st.spinner(" good things take time"):
        st.session_state.vector_index = demo.hr_index()

input_text = st.text_area("Input text", label_visibility="collapsed")
go_button = st.button("Start", type = "primary")

if go_button:
    with st.spinner("Insert a gorgeously stupendously beautiful quote"):
        response_content = demo.hr_rag_response(index = st.session_state.vector_index, question=input_text)
        st.write(response_content)