class PromptHandler:
    def handle_image_placeholders(self, prompt, images):
        """
        Handle image placeholders in the prompt.

        Parameters
        ----------
        prompt : str
            The prompt string.

        Returns
        -------
        str or list
            The prompt string with image placeholders replaced.
        """
        raise NotImplementedError("Subclasses must implement this method.")