import streamlit as st
from src.inference import GDPRInference

# Page config
st.set_page_config(page_title="GDPR Gemma Assistant", page_icon="⚖️")

st.title("⚖️ GDPR Compliance Assistant")
st.markdown("""
Gemma-2B-it 모델을 DPO로 파인튜닝하여 GDPR 준수 관련 질문에 특화된 비서입니다.
""")

# Load model (cached to avoid reloading on every rerun)
@st.cache_resource
def load_gdpr_model():
    return GDPRInference()

with st.spinner("모델을 불러오는 중입니다... 잠시만 기다려 주세요."):
    assistant = load_gdpr_model()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("GDPR에 대해 궁금한 점을 물어보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = assistant.generate(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
st.sidebar.header("About Model")
st.sidebar.info("""
- **Base Model:** Gemma-2B-it
- **Training:** DPO (Direct Preference Optimization)
- **Dataset:** GDPR QA Instruct Dataset
- **Framework:** Transformers, PEFT, TRL
""")
