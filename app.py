import gradio as gr
import os
import sys
import json 
import requests

MODEL = os.getenv("MODEL")
API_URL = os.getenv("API_URL")
DISABLED = os.getenv("DISABLED") == 'True'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NUM_THREADS = int(os.getenv("NUM_THREADS"))

print (NUM_THREADS)

def exception_handler(exception_type, exception, traceback):
    print("%s: %s" % (exception_type.__name__, exception))
sys.excepthook = exception_handler
sys.tracebacklimit = 0

#https://github.com/gradio-app/gradio/issues/3531#issuecomment-1484029099
def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)
    
def predict(inputs, top_p, temperature, chat_counter, chatbot, history, request:gr.Request):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{inputs}"}],
        "temperature" : 1.0,
        "top_p":1.0,
        "n" : 1,
        "stream": True,
        "presence_penalty":0,
        "frequency_penalty":0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    # print(f"chat_counter - {chat_counter}")
    if chat_counter != 0 :
        messages = []
        for i, data in enumerate(history):
            if i % 2 == 0:
                role = 'user'
            else:
                role = 'assistant'
            message = {}
            message["role"] = role
            message["content"] = data
            messages.append(message)
        
        message = {}
        message["role"] = "user" 
        message["content"] = inputs
        messages.append(message)
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature" : temperature,
            "top_p": top_p,
            "n" : 1,
            "stream": True,
            "presence_penalty":0,
            "frequency_penalty":0,
        }

    chat_counter += 1

    history.append(inputs)
    token_counter = 0 
    partial_words = "" 
    counter = 0

    try:
        # make a POST request to the API endpoint using the requests.post method, passing in stream=True
        response = requests.post(API_URL, headers=headers, json=payload, stream=True)
        response_code = f"{response}"
        #if response_code.strip() != "<Response [200]>":
        #    #print(f"response code - {response}")
        #    raise Exception(f"Sorry, hitting rate limit. Please try again later. {response}")
        
        for chunk in response.iter_lines():
            #Skipping first chunk
            if counter == 0:
                counter += 1
                continue
                #counter+=1
            # check whether each line is non-empty
            if chunk.decode() :
                chunk = chunk.decode()
                # decode each line as response data is in bytes
                if len(chunk) > 12 and "content" in json.loads(chunk[6:])['choices'][0]['delta']:
                    partial_words = partial_words + json.loads(chunk[6:])['choices'][0]["delta"]["content"]
                    if token_counter == 0:
                        history.append(" " + partial_words)
                    else:
                        history[-1] = partial_words
                    token_counter += 1
                    yield [(parse_codeblock(history[i]), parse_codeblock(history[i + 1])) for i in range(0, len(history) - 1, 2) ], history, chat_counter, response, gr.update(interactive=False), gr.update(interactive=False)  # resembles {chatbot: chat, state: history}  
    except Exception as e:
        print (f'error found: {e}')
    yield [(parse_codeblock(history[i]), parse_codeblock(history[i + 1])) for i in range(0, len(history) - 1, 2) ], history, chat_counter, response, gr.update(interactive=True), gr.update(interactive=True)
    print(json.dumps({"chat_counter": chat_counter, "payload": payload, "partial_words": partial_words, "token_counter": token_counter, "counter": counter}))
                   

def reset_textbox():
    return gr.update(value='', interactive=False), gr.update(interactive=False)

title = """<h1 align="center">Chat GPT 4o online</h1>"""
if DISABLED:
    title = """<h1 align="center" style="color:red">This app has reached OpenAI's usage limit. We are currently requesting an increase in our quota. Please check back in a few days.</h1>"""
description = """Language models can be conditioned to act like dialogue agents through a conversational prompt that typically takes the form:
```
User: <utterance>
Assistant: <utterance>
User: <utterance>
Assistant: <utterance>
...
```
In this app, you can explore the outputs of a gpt-4 LLM.
"""

theme = gr.themes.Default(primary_hue="green")                

with gr.Blocks(css = """#col_container { margin-left: auto; margin-right: auto;}
                #chatbot {height: 520px; overflow: auto;}""",
              theme=theme) as demo:
    gr.HTML(title)
    gr.HTML("""<h3 align="center">This app provides you full access to Chat GPT-4o from OpenAI thanks to <a href="https://stablediffusion.fr">Stable Diffusion AI online</a>. You don't need any OPENAI API key.<br><br>If this app doesn't respond, it's likely due to too much visitors. Consider trying <a href="https://stablediffusion.fr/chatgpt3">Chat GPT-3.5</a> or <a href="https://stablediffusion.fr/llama2">Llama 2</a> apps.</h3>""")
    with gr.Column(elem_id = "col_container", visible=True) as main_block:
        #openai_api_key = gr.Textbox(type='password', label="Enter only your OpenAI API key here")
        chatbot = gr.Chatbot(elem_id='chatbot') #c
        inputs = gr.Textbox(placeholder= "Hi there!", label= "Type an input and press Enter") #t
        state = gr.State([]) #s
        with gr.Row():
            with gr.Column(scale=7):
                b1 = gr.Button(visible=not DISABLED)
            with gr.Column(scale=3):
                server_status_code = gr.Textbox(label="Status code from OpenAI server", )
    
        #inputs, top_p, temperature, top_k, repetition_penalty
        with gr.Accordion("Parameters", open=False):
            top_p = gr.Slider( minimum=-0, maximum=1.0, value=1.0, step=0.05, interactive=True, label="Top-p (nucleus sampling)",)
            temperature = gr.Slider( minimum=-0, maximum=5.0, value=1.0, step=0.1, interactive=True, label="Temperature",)
            #top_k = gr.Slider( minimum=1, maximum=50, value=4, step=1, interactive=True, label="Top-k",)
            #repetition_penalty = gr.Slider( minimum=0.1, maximum=3.0, value=1.03, step=0.01, interactive=True, label="Repetition Penalty", )
            chat_counter = gr.Number(value=0, visible=False, precision=0)
    

        def enable_inputs():
            return main_block.update(visible=True)

    inputs.submit(reset_textbox, [], [inputs, b1], queue=False)
    inputs.submit(predict, [inputs, top_p, temperature, chat_counter, chatbot, state], [chatbot, state, chat_counter, server_status_code, inputs, b1],)  #openai_api_key
    b1.click(reset_textbox, [], [inputs, b1], queue=False)
    b1.click(predict, [inputs, top_p, temperature, chat_counter, chatbot, state], [chatbot, state, chat_counter, server_status_code, inputs, b1],)  #openai_api_key
             
    demo.queue(max_size=10, api_open=False).launch(share=False)
