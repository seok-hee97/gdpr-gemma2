import streamlit as st
import os
from src.inference import GDPRInference
from src import config

# Page config
st.set_page_config(page_title="GDPR Gemma Assistant", page_icon="⚖️", layout="wide")

st.title("⚖️ GDPR Compliance Assistant")
st.markdown("""
Google's **Gemma-2B-it** 모델을 기반으로 GDPR(유럽 일반 데이터 보호 규칙) 준수를 위해 특화된 AI 어시스턴트입니다.
DPO(Direct Preference Optimization) 기법을 통해 법적 정확성과 전문성을 높였습니다.
""")

# Sidebar: Model Configuration
st.sidebar.header("🛠️ 모델 설정")
model_option = st.sidebar.selectbox(
    "사용할 모델 단계를 선택하세요:",
    ("DPO (최종 모델)", "SFT (지식 학습)", "Base (기본 모델)")
)

# Mapping option to model_path
model_map = {
    "DPO (최종 모델)": "dpo",
    "SFT (지식 학습)": "sft",
    "Base (기본 모델)": "base"
}
selected_stage = model_map[model_option]

# Sidebar: About Model Info
st.sidebar.divider()
st.sidebar.header("📊 Model Info")
st.sidebar.info(f"""
- **Base Model:** `{config.BASE_MODEL_NAME}`
- **Method:** QLoRA + SFT -> DPO
- **Training Stage:** {model_option}
""")

# Load model (cached to avoid reloading on every rerun)
@st.cache_resource(show_spinner=False)
def get_assistant(stage):
    try:
        # 특정 단계의 모델 로드 시도
        return GDPRInference(model_path=stage)
    except Exception as e:
        # 모델 파일이 없는 경우 등에 대한 에러 핸들링
        st.error(f"⚠️ 모델 로드 실패: {str(e)}")
        st.info("💡 학습(Stage 1-3)을 먼저 완료했는지 확인해 주세요. 만약 학습 전이라면 'Base (기본 모델)'을 선택해 보세요.")
        return None

# Load the selected model
assistant = get_assistant(selected_stage)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat if model stage changes
if "last_stage" not in st.session_state:
    st.session_state.last_stage = selected_stage

if st.session_state.last_stage != selected_stage:
    st.session_state.messages = []
    st.session_state.last_stage = selected_stage
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("GDPR에 대해 궁금한 점을 물어보세요."):
    if assistant is None:
        st.warning("모델이 로드되지 않았습니다. 설정을 확인해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"[{model_option}] 답변 생성 중..."):
                try:
                    response = assistant.generate(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"생성 중 오류 발생: {e}")

# Sidebar Footer
st.sidebar.divider()
st.sidebar.caption("© 2024 GDPR-Gemma Project | Powered by Google Gemma")
