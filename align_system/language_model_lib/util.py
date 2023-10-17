import re

def dialog_from_string(dialog_string):
    dialog_markers = {
        '=== system': 'system',
        '=== user': 'user',
        '=== assistant': 'assistant',
    }
    dialog = []
    lines = dialog_string.split('\n')
    current_role = ''
    current_content = ''
    for line in lines:
        if line.strip() in dialog_markers:
            if current_role and current_content:
                dialog.append({
                    'role': current_role,
                    'content': current_content.strip()
                })
            current_role = dialog_markers[line.strip()]
            current_content = ''
        else:
            current_content += f'{line}\n'
    if current_role and current_content:
        dialog.append({
            'role': current_role,
            'content': current_content.strip()
        })
    return dialog

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def format_template(template, **substitutions):
    for key, value in substitutions.items():
        key = '{{%s}}' % key
        if not key in template:
            raise Exception(f'Could not find key {key} in template')
        template = template.replace(key, value)
    
    # ensure there are no strings sorrounded by {{ }}
    matches = re.findall(r'{{.*?}}', template)
    # if there are any matches, raise an exception
    if len(matches) > 0:
        raise Exception(f'Unsubstituited key(s) in template: {matches}')
    
    return template


def extract_kdma_description(descriptions_file):
    kdma_dict = {}
    kdma_name = None
    kdma_description = ''

    with open(descriptions_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if kdma_name is not None:
                    kdma_dict[kdma_name] = kdma_description.strip()
                kdma_name = line[1:].strip()
                kdma_description = ''
            else:
                kdma_description += line

        if kdma_name is not None and kdma_name not in kdma_dict:
            kdma_dict[kdma_name] = kdma_description.strip()

    return kdma_dict


def dialog_to_string(dialog):
    output = ''

    for dialog_piece in dialog:
        role = dialog_piece['role']
        content = dialog_piece['content']
        output += f"=== {role}\n"
        output += f"{content}\n"

    return output