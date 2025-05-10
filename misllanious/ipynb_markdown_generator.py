import json
import os
import argparse
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file (like API keys)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


def load_notebook(file_path: str) -> Dict[str, Any]:
    """Load a Jupyter notebook from a file path and return its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Remove BOM character if present (prevents JSON parsing issues)
        if content.startswith('\ufeff'):
            content = content[1:]
        return json.loads(content)


def extract_code_cells(notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract only non-empty code cells from the notebook."""
    return [
        cell for cell in notebook.get('cells', [])
        if cell.get('cell_type') == 'code' and ''.join(cell.get('source', [])).strip()
    ]


def get_output_text(outputs: List[Dict]) -> str:
    """Extract readable output text from code cell outputs."""
    text = ""
    for output in outputs:
        if 'text' in output:
            t = output['text']
            text += ''.join(t) if isinstance(t, list) else t
        elif 'data' in output and 'text/plain' in output['data']:
            t = output['data']['text/plain']
            text += ''.join(t) if isinstance(t, list) else t
    return text[:997] + "..." if len(text) > 1000 else text  # Truncate if too long


def analyze_code(code: str, cell_number: int, outputs: List[Dict]) -> str:
    """Generate markdown documentation using LLM for a code block."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    output_text = get_output_text(outputs)

    # Craft the prompt for LLM
    prompt = f"""
    Analyze this Python code (Cell #{cell_number}):

    ```python
    {code}
    ```

    {f"The code output was: {output_text}" if output_text else ""}

    Provide a markdown cell explanation with:
    1. High-level summary
    2. Explanation of key variables/functions
    3. Libraries or techniques used
    4. Warnings or edge cases

    Output only markdown suitable for a Jupyter cell.
    """

    # Invoke the LLM with the prompt and return its markdown response
    response = llm.invoke([HumanMessage(content=prompt)]).content
    # Remove markdown formatting if present
    if response.startswith("```markdown"):
        response = response.replace("```markdown", "", 1).strip()
    if response.endswith("```"):
        response = response.rsplit("```", 1)[0].strip()
    return response



def generate_commented_code(code: str) -> str:
    """Generate inline comments for each line of code using LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Craft the prompt to generate comments for each line of code
    prompt = f"""
    Provide inline comments for the following Python code. 
    Comment each line with an explanation of what it does, using Python-style comments (# comment):

    ```python
    {code}
    ```

    Output only the commented code, maintaining the original line structure.
    """
    
    # Invoke the LLM to get commented code
    response = llm.invoke([HumanMessage(content=prompt)]).content
    # Ensure response is formatted correctly (if LLM includes code block formatting)
    return response.replace('```python', '').replace('```', '').strip()



def create_annotated_notebook(input_path: str) -> Dict[str, Any]:
    """Create a new notebook with markdown inserted above each code cell and inline comments added."""
    # Load the original notebook
    original_nb = load_notebook(input_path)

    # Extract only code cells
    code_cells = extract_code_cells(original_nb)

    # Initialize list to hold new cells (markdown + code)
    new_cells = []

    # Loop through each code cell and process it
    for i, cell in enumerate(code_cells, 1):
        code = ''.join(cell.get('source', []))
        outputs = cell.get('outputs', [])

        print(f"Analyzing code block {i}...")  # Log progress

        try:
            markdown = analyze_code(code, i, outputs)
            commented_code = generate_commented_code(code)
        except Exception as e:
            markdown = f"*Error generating markdown for Cell {i}: {e}*"
            commented_code = code  # Fallback to original code if LLM fails

        # Insert markdown explanation BEFORE the code cell
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": markdown.splitlines(keepends=True)  # Preserve line breaks
        })

        # Insert the original code cell with inline comments
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": commented_code.splitlines(keepends=True)  # Insert commented code
        })

    # Build the new notebook structure
    return {
        "cells": new_cells,
        "metadata": original_nb.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5
    }


def save_ipynb(nb_data: Dict[str, Any], output_path: str) -> None:
    """Save the modified notebook to disk as a new .ipynb file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=2)


if __name__ == "__main__":
    # Set up argument parser to receive the notebook path
    parser = argparse.ArgumentParser(description='Insert LLM-generated markdown above each code cell.')
    parser.add_argument('notebook_path', type=str, help='Path to the input .ipynb file')
    args = parser.parse_args()

    # Extract base filename (without extension)
    base_name = os.path.splitext(os.path.basename(args.notebook_path))[0]

    # Generate output file name
    output_path = f"{base_name}_annotated.ipynb"

    print(f"Loading notebook: {args.notebook_path}")
    annotated_nb = create_annotated_notebook(args.notebook_path)
    
    print(f"Saving annotated notebook to: {output_path}")
    save_ipynb(annotated_nb, output_path)

    print("Annotation complete.")
