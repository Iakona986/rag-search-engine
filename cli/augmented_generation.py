from search_utils import load_movies, gemini_call, RRF_K, SEARCH_MULTIPLIER
from hybrid_search import HybridSearch

def rag_command(query):
    movies = load_movies()
    searcher = HybridSearch(movies)
    movies_list = searcher.rrf_search(query)
    context_docs = "\n".join([f"- {m['title']}: {m['document']}" for m in movies_list])
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {context_docs}

                Provide a comprehensive answer that addresses the query:"""
    answer = gemini_call(prompt)
    return movies_list, answer

def summarize_command(query, limit=5):
    movies = load_movies()
    searcher = HybridSearch(movies)
    movies_list = searcher.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)
    context_docs = "\n".join([f"- {m['title']}: {m['document']}" for m in movies_list])
    prompt = f"""
                Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results:
                {context_docs}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
                """
    summary = gemini_call(prompt)
    return movies_list[:limit], summary

def citations_command(query, limit=5):
    movies = load_movies()
    searcher = HybridSearch(movies)
    movies_list = searcher.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)
    context_docs = "\n".join([f"- {m['title']}: {m['document']}" for m in movies_list])
    prompt = f"""Answer the question or provide information based on the provided documents.

                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                    Query: {query}

                    Documents:
                    {context_docs}

                    Instructions:
                    - Provide a comprehensive answer that addresses the query
                    - Cite sources using [1], [2], etc. format when referencing information
                    - If sources disagree, mention the different viewpoints
                    - If the answer isn't in the documents, say "I don't have enough information"
                    - Be direct and informative

                    Answer:"""
    answer = gemini_call(prompt)
    return movies_list[:limit], answer

def question_command(query, limit=5):
    movies = load_movies()
    searcher = HybridSearch(movies)
    movies_list = searcher.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)
    context_docs = "\n".join([f"- {m['title']}: {m['document']}" for m in movies_list])
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {query}

        Documents:
        {context_docs}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:"""
    answer = gemini_call(prompt)
    return movies_list[:limit], answer