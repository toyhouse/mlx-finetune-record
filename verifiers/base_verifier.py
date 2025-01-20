class BaseVerifier:
    """
    Generic interface for a verifier that assigns a reward
    to a generated text.
    """
    def get_reward(self, text: str) -> float:
        raise NotImplementedError("Verifier must implement get_reward method.")