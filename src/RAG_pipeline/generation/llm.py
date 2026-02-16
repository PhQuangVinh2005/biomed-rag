"""Generate responses using an LLM."""
import os
from openai import OpenAI


class LLM:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
