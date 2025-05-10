"""
IPython Notebook Markdown Generator

A simple tool that analyzes .ipynb files and generates markdown documentation 
for each code block using an LLM.
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
    """Load a Jupyter notebook file and return its contents as a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Handle potential JSON issues by checking for BOM and other special characters
            if content.startswith('\ufeff'):  # Remove BOM if present
                content = content[1:]
            notebook = json.loads(content)
        return notebook
    except json.JSONDecodeError as je:
        # Provide specific information about JSON parsing errors
        line_no = je.lineno
        col_no = je.colno
        raise ValueError(f"Invalid JSON in notebook file at line {line_no}, column {col_no}: {je.msg}")
    except Exception as e:
        raise ValueError(f"Error loading notebook file: {str(e)}")


def extract_code_blocks(notebook: Dict[str, Any]) -> List[Dict]:
    """Extract code blocks from a notebook."""
    code_blocks = []
    cell_number = 1
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            code_content = ''.join(cell.get('source', []))
            if code_content.strip():  # Only include non-empty blocks
                code_blocks.append({
                    'cell_number': cell_number,
                    'code': code_content,
                    'outputs': cell.get('outputs', [])
                })
        cell_number += 1
    
    return code_blocks


def get_output_text(outputs: List[Dict]) -> str:
    """Extract readable text from cell outputs."""
    output_text = ""
    for output in outputs:
        if 'text' in output:
            # Handle text being either a string or a list of strings
            text_content = output['text']
            if isinstance(text_content, list):
                output_text += ''.join(text_content)
            else:
                output_text += text_content
        elif 'data' in output:
            if 'text/plain' in output['data']:
                text_plain = output['data']['text/plain']
                if isinstance(text_plain, list):
                    output_text += ''.join(text_plain)
                else:
                    output_text += text_plain
    
    # Truncate very long outputs
    if len(output_text) > 1000:
        output_text = output_text[:997] + "..."
    
    return output_text


def analyze_code_block(code: str, cell_number: int, outputs: List[Dict] = None) -> str:
    """Analyze a single code block using an LLM."""
    # Specify model to ensure compatibility
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    output_text = get_output_text(outputs) if outputs else ""
    
    prompt = f"""
    Analyze the following Python code block (Cell #{cell_number}) from a Jupyter notebook:
    
    ```python
    {code}
    ```
    
    {f"The code generated this output: {output_text}" if output_text else ""}
    
    Generate a detailed markdown documentation for this code block. Your documentation should:
    1. Explain what the code does at a high level
    2. Describe the purpose and functionality of key variables, functions, or algorithms
    3. Highlight any important libraries or techniques used
    4. Include any potential warnings, edge cases, or optimization notes if relevant
    
    Format your response as markdown that would be suitable for inclusion in a Jupyter notebook.
    Only include the markdown text without any additional explanation or meta-commentary.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_markdown_documentation(notebook_path: str) -> str:
    """Generate markdown documentation for all code blocks in a notebook."""
    try:
        # Load the notebook and extract code blocks
        notebook = load_notebook(notebook_path)
        code_blocks = extract_code_blocks(notebook)
        
        if not code_blocks:
            return "# No code blocks found in the notebook"
        
        # Generate documentation
        final_markdown = "# Jupyter Notebook Code Documentation\n\n"
        
        print(f"Processing {len(code_blocks)} code blocks...")
        
        for idx, block in enumerate(code_blocks):
            print(f"Analyzing block {idx+1}/{len(code_blocks)}...")
            
            try:
                analysis = analyze_code_block(
                    code=block['code'],
                    cell_number=block['cell_number'],
                    outputs=block.get('outputs', [])
                )
                
                final_markdown += f"## Cell {block['cell_number']}\n\n"
                final_markdown += f"```python\n{block['code']}\n```\n\n"
                final_markdown += f"{analysis}\n\n"
                final_markdown += "---\n\n" if idx < len(code_blocks) - 1 else ""
            except Exception as cell_error:
                print(f"Error processing cell {block['cell_number']}: {str(cell_error)}")
                final_markdown += f"## Cell {block['cell_number']}\n\n"
                final_markdown += f"```python\n{block['code']}\n```\n\n"
                final_markdown += f"*Error analyzing this code block: {str(cell_error)}*\n\n"
                final_markdown += "---\n\n" if idx < len(code_blocks) - 1 else ""
        
        return final_markdown
        
    except Exception as e:
        import traceback
        return f"# Error processing notebook\n\nAn error occurred: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def save_markdown(markdown_content: str, output_path: str) -> None:
    """Save the generated markdown to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate markdown documentation for Jupyter notebook code blocks.')
    parser.add_argument('notebook_path', type=str, help='Path to the .ipynb file')
    
    args = parser.parse_args()
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(args.notebook_path))[0]
    
    # Hard-coded output filename pattern with .md extension (not .ipynb)
    output_path = f"{base_name}_markdownformat.md"
    
    print(f"Processing notebook: {args.notebook_path}")
    markdown_content = generate_markdown_documentation(args.notebook_path)
    
    save_markdown(markdown_content, output_path)
    print(f"Documentation generated successfully and saved to {output_path}")