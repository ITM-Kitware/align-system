import jinja2
import os
import logging

# Load templates in the templates directory
logging.info('Loading persona ADM templates from %s', os.path.join(os.path.abspath(os.path.dirname(__file__)), 'templates'))
template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'templates'))
template_env = jinja2.Environment(loader=template_loader)

# Load the template for the persona ADM
probe_template = template_env.get_template('probe.jinja')

__all__ = ['probe_template']
