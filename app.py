import gradio as gr
from http import HTTPStatus
import dashscope
import json
import os

# Preset messages and functions (common for all models)
preset_messages = {
    "Greeting": '[{"role": "user", "content": "Hello!"}]',
    # Add more presets here
}

def load_presets(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return preset_messages

def save_preset(name, content, file_path):
    presets = load_presets(file_path)
    presets[name] = content
    with open(file_path, "w") as file:
        json.dump(presets, file)

def get_preset_content(preset_name):
    presets = load_presets("presets.json")
    return presets.get(preset_name, "")

# def load_env():
    # with open('.env', 'r') as file:
        # for line in file:
            # if line.startswith('#') or not line.strip():
                # continue
            # key, value = line.strip().split('=', 1)
            # os.environ[key] = value

# load_env()

# dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

# Specific functions for Tongyi Qianwen
def call_tongyi_qianwen_with_messages(model_name, user_message, assistant_message, max_tokens, temperature, top_p):
    messages = []
    if user_message:
        messages.append({"role": "user", "content": user_message})
    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message})

    response = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        result_format='message'
    )

    if response.status_code == HTTPStatus.OK:
        return str(response)
    else:
        return f'Error: {response.message}'

def call_tongyi_qianwen_with_prompt(model_name, prompt, max_tokens, temperature, top_p):
    response = dashscope.Generation.call(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    print("Full Response:", response)

    if response.status_code == HTTPStatus.OK:
        response_text = response.output.get('text', 'No text found in response')
        return response_text
    else:
        return f'Error: {response.message}'

# Gradio interface setup
with gr.Blocks() as app:
    gr.Markdown("### Advanced Model Selection Menu")

    model_selector = gr.Dropdown(label="Select a Model", choices=[
        "通义千问", "LLaMa2 大语言模型", "通义千问开源系列", "百川开源大语言模型", 
        "通义万相系列", "FaceChain人物写真生成", "WordArt锦书-创意文字生成", 
        "通用文本向量", "ONE-PEACE多模态向量表征", "StableDiffusion文生图模型", 
        "OpenNLU开放域文本理解模型", "语音合成", "ChatGLM开源双语对话语言模型", 
        "智海三乐教育大模型", "Paraformer语音识别", "姜子牙通用大模型", 
        "Dolly开源大语言模型", "BELLE开源中文对话大模型", "MOSS开源对话语言模型", 
        "元语功能型对话大模型V2", "BiLLa开源推理能力增强模型"
    ])


    with gr.Tabs() as tabs:
        with gr.Tabs("通义千问"):

            tongyi_model_selector = gr.Dropdown(label="Select Tongyi Model", choices=[
                "qwen_turbo", "qwen-plus", "qwen-max", "qwen-max-1201", "qwen-max-longcontext"
            ])

            with gr.Column():
                user_message_input = gr.Textbox(label="Your Message", placeholder="Type your message here")
                assistant_message_input = gr.Textbox(label="Assistant's Reply (if any)", placeholder="Type assistant's reply here")
                max_tokens_input_msg = gr.Slider(minimum=10, maximum=2000, step=10, value=1500, label="Max Tokens")
                temperature_input_msg = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Temperature")
                top_p_input_msg = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="Top P")
                submit_button_msg = gr.Button("Submit")
                output_msg = gr.Textbox(label="Model Response")

            submit_button_msg.click(
                call_tongyi_qianwen_with_messages,
                inputs=[tongyi_model_selector, user_message_input, assistant_message_input, max_tokens_input_msg, temperature_input_msg, top_p_input_msg],
                outputs=output_msg
            )

            with gr.Column():
                prompt_input = gr.Textbox(label="Enter Prompt")
                max_tokens_input_prompt = gr.Slider(minimum=10, maximum=2000, step=10, value=1500, label="Max Tokens")
                temperature_input_prompt = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Temperature")
                top_p_input_prompt = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="Top P")
                submit_button_prompt = gr.Button("Submit")
                output_prompt = gr.Textbox(label="Model Response")

            submit_button_msg.click(
                call_tongyi_qianwen_with_messages,
                inputs=[tongyi_model_selector, user_message_input, assistant_message_input, max_tokens_input_msg, temperature_input_msg, top_p_input_msg],
                outputs=output_msg
            )

        # with gr.Tab("LLaMa2 大语言模型"):
            # LLaMa2 大语言模型的交互设置
            # ...

        # Add other tabs for different models

    model_selector.change(lambda model: tabs.set_tab(model), inputs=model_selector, outputs=tabs)

app.launch()
