

# Gradio Apps

## Introduction
------------

This repo explores building apps with Gradio. 

Explores: 

- Text summarizer app that uses the Distill Bart CNN model

text and the second recognizes named entities. 
- An image captioning app with BLIP (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation) from Salesforce
- An image generation app that generates images from input text using Stable Diffusion v.1.5
- A game application that creates an caption from and image and new image from that caption. 
- Chat with Falcon 40B, and opensource LLM (Note: Free inferencing endpiong for Falcon40B instruct has been disconnected from HF)

Gradio is sick! 


# Dependencies 
----------------------------
Install:

1. Clone the repository to your local machine.

2. Create VM and install dependencies from .yml

Using `micromamba`:
``` bash
cd gradio_apps
micromamba env create -f environment.yml
micromamba activate llm-deep
```

3. Create a `.env` file in the root directory of the project. Inside the file, add your OpenAI API key:

```makefile
WEAVIATE_API_KEY="your_api_key_here"
COHERE_API_KEY = "your_api_key_here"
OPENAI_API_KEY= "your_api_key_here"
HF_API_KEY= "your_api_key_here"
```

