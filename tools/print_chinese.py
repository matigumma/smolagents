from transformers import tool

@tool
def run(**input: str) -> str:
    """
        Prints the Chinese input.
        Args:
          input: The chinese string to print out.
    """
    
    print(f"Translation: {input['input']}")
    return input