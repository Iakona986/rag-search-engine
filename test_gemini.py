import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError ("No API key set")

client = genai.Client(api_key=api_key)

def main():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, default="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.", help="User prompt")
    #parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]
    # Execution loop (limited to 20 turns)
    for iteration in range(20):
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=messages,
        )
        if not response.candidates:
            print("Error: The model failed to generate a response.")
            break
        messages.append(response.candidates[0].content)
        function_calls = [part.function_call for part in response.candidates[0].content.parts if part.function_call]

        if not function_calls:
            if response.text:
                print(f"\nAI: {response.text}")
                print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
                print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
            return

        function_responses = []

        for call in function_calls:
            result_content = call_function(call, verbose=args.verbose)
            function_responses.append(result_content.parts[0])
        messages.append(types.Content(role="user", parts=function_responses))
    print(f"\nError: The agent reached the maximum iteration limit ({iteration + 1}).")
    if args.verbose == True:
        print(f"User prompt: {args.user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

    function_results_parts = []

    if response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                # Execute the function using our dispatcher
                function_call_result = call_function(part.function_call, verbose=args.verbose)

                # Validation Checks
                if not function_call_result.parts:
                    raise Exception("Function call returned no parts.")

                resp_part = function_call_result.parts[0]
                if resp_part.function_response is None:
                    raise Exception("Part does not contain a function_response.")

                if resp_part.function_response.response is None:
                    raise Exception("FunctionResponse contains no data in .response field.")

                # Store the part for the LLM's next turn
                function_results_parts.append(resp_part)

                if args.verbose:
                    print(f"-> {resp_part.function_response.response}")

            elif part.text:
                print(part.text)


if __name__ == "__main__":
    main()
