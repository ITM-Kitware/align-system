import re


class Dialog:
    
    @classmethod
    def from_string(cls, string):
        dialog_markers = {
            '=== system': 'system',
            '=== user': 'user',
            '=== assistant': 'assistant',
        }
        dialog = []
        lines = string.split('\n')
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

    
    @classmethod
    def from_template(cls, template, **substitutions):
        cls.from_string(format_template(template, **substitutions))
    
    
    def __init__(self, messages):
        self.messages = messages
    
    
    def __str__(self):
        output = ''

        for dialog_piece in self.messages:
            role = dialog_piece['role']
            content = dialog_piece['content']
            output += f"=== {role}\n"
            output += f"{content}\n"

        return output


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
        raise Exception(f'Could not find values for {matches} in template')
    
    return template