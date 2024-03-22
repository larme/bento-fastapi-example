import uuid
from typing import AsyncGenerator

import bentoml
import fastapi
from annotated_types import Ge, Le
from typing_extensions import Annotated

MAX_TOKENS = 512
MODEL_ID = "facebook/opt-350m"

app = fastapi.FastAPI()


@bentoml.mount_asgi_app(app)
@bentoml.service(
    name="bentovllm-opt-completion-service",
    traffic={"timeout": 300},
    resources={"gpu": 1},
)
class VLLM:

    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID, max_model_len=MAX_TOKENS
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)
        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)

    @app.get("/")
    def root(self):
        return fastapi.responses.HTMLResponse(ROOT_TEMPLATE)

    @app.get("/model_card")
    def model_card(self):
        return fastapi.responses.JSONResponse(
            {
                "model_id": MODEL_ID,
                "description": "OpenAI's GPT-3 model fine-tuned on the OpenWebText dataset",
                "license": "MIT",
                "author": "OpenAI",
            }
        )


ROOT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stream Response with Prompt</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        #response {
            white-space: pre-wrap;
            background-color: #f0f0f0;
            padding: 10px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <label for="prompt">Enter Prompt:</label>
    <input type="text" id="prompt" name="prompt">
    <button id="submit">Send Request</button>

    <div id="response"></div>

    <script>
        document.getElementById('submit').addEventListener('click', function() {
            var promptContent = document.getElementById('prompt').value;
            var responseContainer = document.getElementById('response');
            responseContainer.textContent = ''; // Clear previous response

            // Fetch API to send a POST request
            fetch('generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({prompt: promptContent})
            }).then(function(response) {
                // Stream the response
                const reader = response.body.getReader();
                return new ReadableStream({
                    start(controller) {
                        function push() {
                            reader.read().then(({done, value}) => {
                                if (done) {
                                    controller.close();
                                    return;
                                }
                                // Convert the Uint8Array to string
                                const decoder = new TextDecoder();
                                const text = decoder.decode(value);
                                responseContainer.textContent += text; // Append the text to the container
                                push();
                            });
                        }
                        push();
                    }
                });
            }).then(stream => new Response(stream))
            .then(response => response.text())
            .then(text => console.log(text))
            .catch(err => console.error('Fetch error:', err));
        });
    </script>
</body>
</html>
"""
