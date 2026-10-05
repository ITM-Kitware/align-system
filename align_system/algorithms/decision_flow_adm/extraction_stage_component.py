import json
from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.decision_flow_adm.utils import validate_structured_response
from align_system.exceptions import SceneSkipException

log = logging.getLogger(__name__)


class ExtractionStageComponent(ADMComponent):
    def __init__(
        self,
        structured_inference_engine,
        scenario_description_template,
        system_prompt_template,
        prompt_template,
        output_schema_template,
        max_json_retries=5,
        **kwargs,
    ):
          self.structured_inference_engine = structured_inference_engine
          self.scenario_description_template = scenario_description_template
          self.system_prompt_template = system_prompt_template
          self.prompt_template = prompt_template
          self.output_schema_template = output_schema_template
          self.max_json_retries = max_json_retries

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
                                               content=system_prompt))

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
                                    content=prompt))

        output_schema = call_with_coerced_args(
            self.output_schema_template,
            {})

        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

        # Retry loop for structured inference with validation
        response = None
        last_error = None

        for attempt in range(self.max_json_retries):
            try:
                # Run structured inference
                raw_response = self.structured_inference_engine.run_inference(
                    dialog_prompt,
                    output_schema
                )

                # Validate response
                response = validate_structured_response(raw_response)

                # Success - break out of retry loop
                log.info(f"Extraction stage inference succeeded on attempt {attempt + 1}")
                break

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                last_error = e
                log.warning(
                    f"Extraction stage JSON decode error on attempt {attempt + 1}/{self.max_json_retries}: {e}"
                )

                if attempt < self.max_json_retries - 1:
                    log.info("Retrying Extraction stage inference...")
                else:
                    log.error(f"Extraction stage failed after {self.max_json_retries} attempts")
                    raise SceneSkipException(
                        f"Failed to generate valid JSON after {self.max_json_retries} attempts. "
                        f"Last error: {last_error}",
                        component_name="ExtractionStageComponent",
                        last_error=last_error
                    ) from last_error

        log.info(f"Extraction completed: {response['information']}")
        return response['information']
