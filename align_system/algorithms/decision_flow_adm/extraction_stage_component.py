from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)


class ExtractionStageComponent(ADMComponent):
    def __init__(
        self,
        structured_inference_engine,
        scenario_description_template,
        system_prompt_template,
        prompt_template,
        output_schema_template,
        **kwargs,
    ):
          self.structured_inference_engine = structured_inference_engine
          self.scenario_description_template = scenario_description_template
          self.system_prompt_template = system_prompt_template
          self.prompt_template = prompt_template
          self.output_schema_template = output_schema_template

    def run_returns(self):
        return "extraction"

    def run(self, scenario_state, choices, variables=None, **kwargs):
        """Extract key information and context from variables and scenario"""

        dialog = []
        if self.system_prompt_template is not None:
                system_prompt = call_with_coerced_args(
                    self.system_prompt_template, {},
                )

                dialog.insert(0, DialogElement(role='system',
                                               content=system_prompt,
                                               tags=['decisionflow_system_prompt']))

        log.info(type(self.prompt_template))
        scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state})
        log.info(scenario_description)
        prompt = call_with_coerced_args(
            self.prompt_template,
            {'scenario_description': scenario_description, 'choices': choices, 'variables': variables},
        )
        log.info(prompt)
        dialog.append(DialogElement(role='user',
                                    content=prompt,
                                    tags=['decisionflow_extraction']))

        output_schema = call_with_coerced_args(
            self.output_schema_template,
            {})

        response = self.structured_inference_engine.run_inference(
            self.structured_inference_engine.dialog_to_prompt(dialog),
            output_schema
        )

        log.info(f"Extraction completed: {response['information']}")
        return response['information']