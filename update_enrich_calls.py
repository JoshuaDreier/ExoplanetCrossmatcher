import os
import json

def process_file(filepath):
    if filepath.endswith('.py'):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = update_content(content)
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
            
    elif filepath.endswith('.ipynb'):
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            
        changed = False
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                new_source = []
                for line in cell['source']:
                    # Simple heuristic: if the line has .enrich( and it closes on the same line, just replace it
                    # If it's multi-line, it's harder. Let's do a full text update on the joined cell source.
                    pass
                
                full_source = "".join(cell['source'])
                new_full_source = update_content(full_source)
                if new_full_source != full_source:
                    # split back into lines, keeping newlines
                    lines = new_full_source.splitlines(keepends=True)
                    cell['source'] = lines
                    changed = True
                    
        if changed:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
                f.write("\n") # Jupyter format
            print(f"Updated {filepath}")

def update_content(text):
    # Find all occurrences of '.enrich('
    result = text
    offset = 0
    while True:
        idx = result.find('.enrich(', offset)
        if idx == -1:
            break
            
        # Find the matching closing parenthesis
        paren_count = 0
        end_idx = -1
        in_string = False
        string_char = ''
        escape = False
        
        for i in range(idx + 7, len(result)):
            char = result[i]
            
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == string_char:
                    in_string = False
                continue
                
            if char in ["'", '"']:
                in_string = True
                string_char = char
            elif char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    end_idx = i
                    break
                    
        if end_idx != -1:
            # Check if it already has [0]
            if end_idx + 1 < len(result) and result[end_idx+1:end_idx+4] == '[0]':
                offset = end_idx + 4
            else:
                result = result[:end_idx+1] + '[0]' + result[end_idx+1:]
                offset = end_idx + 4
        else:
            offset = idx + 8
            
    return result

for root, dirs, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py') or file.endswith('.ipynb'):
            process_file(os.path.join(root, file))

for root, dirs, files in os.walk('notebooks'):
    for file in files:
        if file.endswith('.py') or file.endswith('.ipynb'):
            process_file(os.path.join(root, file))
