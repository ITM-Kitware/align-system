import re
import os
from typing import List, Dict


def dialog_from_string(dialog_string: str) -> List[Dict[str, str]]:
    """
    Transforms the dialog in string format to a list of dictionary format.

    :param dialog_string: Dialog in string format.
    :return: Dialog in the list of dictionary format.
    """
    # Dictionary to map string markers to role names
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
        if line.strip() in dialog_markers:  # If a line indicates a role change
            if current_role and current_content:  # Save the previous role's dialog
                dialog.append({
                    'role': current_role,
                    'content': current_content.strip()
                })
            current_role = dialog_markers[line.strip()]  # Set the new role
            current_content = ''
        else:  # Continue appending content if the role hasn't changed
            current_content += f'{line}\n'
    # Append the last piece of dialog
    if current_role and current_content:
        dialog.append({
            'role': current_role,
            'content': current_content.strip()
        })
    return dialog


def dialog_to_string(dialog: List[Dict[str, str]]) -> str:
    """
    Transforms the dialog in list of dictionary to string format.

    :param dialog: Dialog in list of dictionary format.
    :return: Dialog in string format.
    """
    output = ''
    for dialog_piece in dialog:
        role = dialog_piece['role']
        content = dialog_piece['content']
        output += f"=== {role}\n"
        output += f"{content}\n"

    return output


def format_template(template: str, **substitutions: str) -> str:
    """
    Replaces placeholders in a template with provided substitutions.

    :param template: The template with placeholders indicated as {{placeholder}}.
    :param substitutions: The substitutions to replace in the template.
    :return: The template with all placeholders substituted.
    """
    for key, value in substitutions.items():
        key = '{{%s}}' % key
        if not key in template:
            raise Exception(f'Could not find key {key} in template')
        template = template.replace(key, value)
    
    # ensure there are no strings surrounded by {{ }}
    matches = re.findall(r'{{.*?}}', template)
    # if there are any matches, raise an exception
    if len(matches) > 0:
        raise Exception(f'Unsubstituited key(s) in template: {matches}')
    
    return template


def read_template(template_file_name: str, template_dir='templates') -> str:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_directory, template_dir, template_file_name)
    
    with open(full_path, 'r') as template_file:
        template = template_file.read()
    
    return template