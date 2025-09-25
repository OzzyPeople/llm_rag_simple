from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError
from typing import Any, Dict, Optional


class GeminiClient:
    def __init__(self, api_key: str, system_prompt: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=200,
        )

    def generate(
            self,
            user_prompt: str,
            response_schema: Optional[BaseModel] = None,
            **kwargs
    ) -> Optional[Dict[str, Any] | str]:
        """
        Generate text from Gemini model using LangChain.

        Args:
            user_prompt: The user's input prompt
            response_schema: Optional Pydantic model for structured output
            **kwargs: Overrides for model parameters

        Returns:
            Dict if response_schema provided and valid, otherwise str
        """
        llm = self.llm.bind(**kwargs) if kwargs else self.llm

        try:
            # Include system prompt in the message chain
            messages = [
                ("system", self.system_prompt),
                ("user", user_prompt)
            ]

            resp = llm.invoke(messages)

            if response_schema:
                try:
                    parsed = response_schema.model_validate_json(resp.content)
                    return parsed.model_dump()
                except ValidationError as e:
                    print(f"[GeminiClient] Schema validation failed: {e}")
                    # Optionally return raw content or raise exception
                    return resp.content

            return resp.content

        except Exception as e:
            print(f"[GeminiClient] Generation failed: {e}")
            return None

    # Optional: Add a method specifically for structured outputs
    def generate_structured(self, user_prompt: str, response_schema: BaseModel, **kwargs):
        """Convenience method for structured outputs only"""
        return self.generate(user_prompt, response_schema, **kwargs)

