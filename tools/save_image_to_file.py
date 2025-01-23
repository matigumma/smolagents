from transformers import tool

@tool
def run(**input: str) -> str:
    """
        Save the image content to a file and finish the task
        Args:
            input: 
                type: image
                description: The image generated from image_generator()
    """
    input['input'].save("imagen.jpg")
    return f"final_answer('File saved as imagen.jpg')\n"