# Class for NNsight-supported language models
from nnsight import LanguageModel, CONFIG

from language_models.model import Model
from language_models.utils import add_retries, limiter


class NNsightModel(Model):
    def __init__(self, name, api_key="b44d4f0a-0e50-4652-800d-b9f77719042e", temperature=0.7, max_tokens=256, device_map='auto', remote=False):
        """
        Args:
            name: name of the model (e.g., "openai-community/gpt2")
            api_key: NNsight API key (optional if already set globally)
            temperature: temperature parameter of model
            max_tokens: maximum number of tokens to generate
            device_map: device mapping for model (e.g., 'auto', 'cuda', 'cpu')
            remote: whether to use remote NDIF execution
        """
        super().__init__(name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.remote = remote
        
        # Set API key if provided
        if api_key:
            CONFIG.set_default_api_key(api_key)
        
        # Load model - parameters aren't loaded locally when using remote=True
        self.model = LanguageModel(name, device_map=device_map)
    
    @add_retries
    @limiter.ratelimit('identity', delay=True)
    def generate_response(self, prompt, n_completions=1):
        """
        Generates a response to a prompt.
        Args:
            prompt: prompt to generate a response to
            n_completions: number of completions to generate
        Returns:
            response: list of responses to the prompt
        """
        responses = []
        
        for _ in range(n_completions):
            with self.model.generate(
                prompt, 
                max_new_tokens=self.max_tokens, 
                temperature=self.temperature,
                remote=self.remote
            ) as generator:
                # Save the generated output
                output_ids = self.model.generator.output.save()
            
            # Decode the generated tokens to text
            # output_ids is a tensor of token IDs
            generated_text = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text to get only the completion
            if generated_text.startswith(prompt):
                completion = generated_text[len(prompt):].strip()
            else:
                # Fallback: just use the generated text as-is
                completion = generated_text.strip()
            
            responses.append(completion)
        
        return responses