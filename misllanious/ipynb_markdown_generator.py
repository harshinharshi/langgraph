"""
IPython Notebook Annotator (Markdown Above Code)

Generates a new notebook with markdown cells (LLM-generated) inserted above each code cell.
"""

import json
import os
import argparse
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def load_notebook(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if content.startswith('\ufeff'):
            content = content[1:]
        return json.loads(content)


def extract_code_cells(notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        cell for cell in notebook.get('cells', [])
        if cell.get('cell_type') == 'code' and ''.join(cell.get('source', [])).strip()
    ]


def get_output_text(outputs: List[Dict]) -> str:
    text = ""
    for output in outputs:
        if 'text' in output:
            t = output['text']
            text += ''.join(t) if isinstance(t, list) else t
        elif 'data' in output and 'text/plain' in output['data']:
            t = output['data']['text/plain']
            text += ''.join(t) if isinstance(t, list) else t
    return text[:997] + "..." if len(text) > 1000 else text


def analyze_code(code: str, cell_number: int, outputs: List[Dict]) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    output_text = get_output_text(outputs)
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
    return llm.invoke([HumanMessage(content=prompt)]).content


def create_annotated_notebook(input_path: str) -> Dict[str, Any]:
    original_nb = load_notebook(input_path)
    code_cells = extract_code_cells(original_nb)

    new_cells = []
    for i, cell in enumerate(code_cells, 1):
        code = ''.join(cell.get('source', []))
        outputs = cell.get('outputs', [])

        try:
            markdown = analyze_code(code, i, outputs)
        except Exception as e:
            markdown = f"*Error generating markdown for Cell {i}: {e}*"

        # First, add markdown explanation
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": markdown.splitlines(keepends=True)
        })

        # Then add the original code cell
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True)
        })

    # Wrap up notebook structure
    return {
        "cells": new_cells,
        "metadata": original_nb.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5
    }


def save_ipynb(nb_data: Dict[str, Any], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Insert LLM markdown cells above each code cell in a notebook.')
    parser.add_argument('notebook_path', type=str, help='Path to the .ipynb file')
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.notebook_path))[0]
    output_path = f"{base_name}_annotated.ipynb"

    print(f"Processing: {args.notebook_path}")
    annotated_nb = create_annotated_notebook(args.notebook_path)
    save_ipynb(annotated_nb, output_path)
    print(f"Saved annotated notebook as {output_path}")
