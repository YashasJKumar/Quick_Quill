import time
import streamlit as st
from groq import Groq
from typing import Generator

st.set_page_config(
    page_icon="🚩",
    layout="wide",
    page_title="Quick Quill"
)

client = Groq(
    api_key="YOUR_API_KEY",
)

# Define the gradient style
gradient_style = """
<style>
.header {
    font-size: 60px;
    font-weight: bold;
    background: -webkit-linear-gradient(yellow, red);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
</style>
"""

# Insert the gradient style into the Streamlit app
st.markdown(gradient_style, unsafe_allow_html=True)


def text_glorifier(text: str, font_size: int):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: {font_size}px; line-height: 1">{text}</span>',
        unsafe_allow_html=True,
    )


st.markdown('<h6 class="header">Quick Quill</h6>', unsafe_allow_html=True)
# text_glorifier("Groq Chat", 60)
st.subheader("", divider="orange", anchor=False)

LLM_OPTIONS = {
    "Gemma 2 9B": "gemma2-9b-it",
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3.1 405B": "llama-3.1-405b-reasoning",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768"
}

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

with st.sidebar:
    st.write("")
    st.image("./groq.png", width=210, use_column_width=False)
    st.write("")
    st.subheader(":blue[Experience Lightning Fast Inference]")
    st.write("")
    st.write("Swiftly produce high-quality responses, much like how a skilled writer would quickly draft eloquent "
             "prose.")
    st.write("")
    st.markdown(":red[Disclaimer]")
    st.write("Run Open Source LLM's running on the fastest inferencing engine powered by GroqCloud")
    selected_llm = st.selectbox(label="Select your LLM here 👇", label_visibility="visible",
                                options=LLM_OPTIONS)
    st.write("")
    temperature_value = st.slider(label="Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5,
                                  help="Choose a temperature value for the creative response of the Model")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '😎'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Enter a prompt"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    start_time = time.time()

    with st.chat_message("user", avatar='😎'):
        st.markdown(prompt)

    try:
        streamer = client.chat.completions.create(
            model=LLM_OPTIONS[selected_llm],
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            temperature=temperature_value,
            max_tokens=1024,
            stream=True,
            top_p=1
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Generating response..."):
                chat_responses_generator = generate_chat_responses(streamer)
                full_response = st.write_stream(chat_responses_generator)
        st.sidebar.write(":green[Response Time: ]" + f" {round(time.time() - start_time, 2)} s.")

    except Exception as e:
        st.error(e, icon="‼️")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
