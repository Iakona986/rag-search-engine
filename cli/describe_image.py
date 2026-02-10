from search_utils import gemini_image_call

def describe_image(image_path: str, query: str) -> str:
    prompt = f"""
             Given the included image and text query, rewrite the text query to improve search results from a movie database.  Make sure to:
             - Synthesize visual and textual information
             - Focus on movie-specific details (actors, scenes, style, etc.)
             - Return only the rewritten query, without any additional commentary.

             Image: {image_path}
             Query: {query}
             """
    return gemini_image_call(prompt, image_path, query)

