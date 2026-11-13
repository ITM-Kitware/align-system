import json
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from align_system.utils import logging
from align_system.utils.version import get_version


log = logging.getLogger(__name__)


def _source_paths(source_run):
    source = Path(to_absolute_path(source_run)).resolve()
    if source.is_dir():
        return source, source / 'input_output.json'
    return source.parent, source


def prepare_replay_config(cfg):
    source_dir, input_output_path = _source_paths(cfg.replay_run)
    output_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    if output_dir == source_dir:
        raise ValueError('Replay output directory must differ from the source run')
    if not input_output_path.is_file():
        raise FileNotFoundError(input_output_path)

    source_cfg = OmegaConf.load(source_dir / '.hydra' / 'config.yaml')
    # Discard the original models and agent tools before recursive instantiation.
    del source_cfg['driver']
    source_cfg.pop('adm', None)
    cfg = OmegaConf.merge(source_cfg, cfg)
    cfg.replay_run = str(input_output_path)

    # Hydra saved the overrides before the source settings were loaded. Save the
    # effective config so this run is self-contained and can itself be replayed.
    hydra_subdir = HydraConfig.get().output_subdir
    if hydra_subdir is not None:
        OmegaConf.save(cfg, output_dir / hydra_subdir / 'config.yaml')
    return cfg


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value.to_dict() if hasattr(value, 'to_dict') else value._asdict()


def _check_equal(expected, actual, context):
    if expected == actual:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f'Replay mismatch at {context}: different fields '
                             f'(expected {list(expected)}, got {list(actual)})')
        for key in expected:
            _check_equal(expected[key], actual[key], f'{context}.{key}')
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f'Replay mismatch at {context}: expected '
                             f'{len(expected)} items, got {len(actual)}')
        for i, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _check_equal(expected_item, actual_item, f'{context}[{i}]')
    else:
        raise ValueError(f'Replay mismatch at {context}: '
                         f'expected {expected!r}, got {actual!r}')


class ReplayDriver:
    def __init__(self, source_run):
        self.source_dir, input_output_path = _source_paths(source_run)
        with input_output_path.open() as f:
            self.records = json.load(f)
        if not isinstance(self.records, list):
            raise ValueError('Replay input_output must be a list of decisions')
        for i, record in enumerate(self.records):
            for field in ('scenario_id', 'full_state', 'alignment_target_id'):
                if field not in record['input']:
                    raise ValueError(f'Replay decision {i} is missing input.{field}')
            if not isinstance(record['output'].get('action'), dict):
                raise ValueError(f'Replay decision {i} is missing output.action')

    def drive(self, cfg):
        interface = cfg.interface
        output_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
        if output_dir == self.source_dir:
            raise ValueError('Replay output directory must differ from the source run')

        meta = {'version': get_version(), 'driver': 'replay',
                'source_run': str(self.source_dir)}
        username = getattr(interface, 'username', None)
        if username is not None:
            meta['username'] = username
        with (output_dir / 'meta.json').open('w') as f:
            json.dump(meta, f, indent=2)

        replayed = []
        scores = []
        with (output_dir / 'input_output.json').open('w') as f:
            json.dump(replayed, f, indent=2)

        index = 0
        while scenario := interface.start_scenario():
            if scenario.id() == '':
                break
            log.info(f'Replaying scenario {scenario.id()}')
            state = scenario.get_state()
            if 'alignment_target' in cfg:
                target = cfg.alignment_target
            elif cfg.align_to_target:
                target = scenario.get_alignment_target()
            else:
                target = None
            target_id = target.id if target is not None and cfg.align_to_target else None

            if target is not None:
                target_dict = _as_dict(target)
                target_path = self.source_dir / 'targets' / f'{target.id}.json'
                if target_path.is_file():
                    with target_path.open() as f:
                        _check_equal(json.load(f), target_dict,
                                     f'decision {index} (scenario {scenario.id()}) alignment_target')
                if cfg.save_alignment_targets:
                    (output_dir / 'targets').mkdir(exist_ok=True)
                    with (output_dir / 'targets' / f'{target.id}.json').open('w') as f:
                        json.dump(target_dict, f, indent=2)

            while not state.scenario_complete:
                context = f'decision {index} (scenario {scenario.id()})'
                if index >= len(self.records):
                    raise ValueError(f'Replay exhausted before {context}; '
                                     'the recording may be incomplete')
                record = self.records[index]
                _check_equal(record['input']['scenario_id'], scenario.id(),
                             f'{context}.scenario_id')
                _check_equal(record['input']['alignment_target_id'], target_id,
                             f'{context}.alignment_target_id')
                _check_equal(record['input']['full_state'], _as_dict(state),
                             f'{context}.full_state')

                recorded_action = record['output']['action']
                available_actions = scenario.get_available_actions()
                action = next((a for a in available_actions
                               if a.action_id == recorded_action.get('action_id')), None)
                if action is None:
                    raise ValueError(f'Replay action {recorded_action.get("action_id")!r} '
                                     f'is unavailable at {context}')
                # Use the interface's action model, including recorded parameters
                # supplied by the ADM or by action expansion in the original driver.
                action = type(action)(**recorded_action)
                log.info(f'Replaying decision {index}: {recorded_action}')
                if recorded_action.get('intent_action', False):
                    state = scenario.intend_action(action)
                else:
                    state = scenario.take_action(action)

                replayed.append(record)
                with (output_dir / 'input_output.json').open('w') as f:
                    json.dump(replayed, f, indent=2)
                index += 1

            if cfg.get('save_last_unstructured_state_per_scenario', False):
                final_target = target if target is not None else scenario.get_alignment_target()
                final_target_id = final_target.id if final_target is not None else None
                final_path = output_dir / f'{scenario.id()}.{final_target_id}.final_state_unstructured.json'
                with final_path.open('w') as f:
                    print(state.unstructured, file=f)

            if target is not None:
                try:
                    score = interface.get_session_alignment(target)
                except Exception:
                    log.exception("Couldn't get session alignment from interface")
                    score = None
                if score is not None:
                    scores.append(_as_dict(score))
                    if cfg.save_scoring_output:
                        with (output_dir / 'scores.json').open('w') as f:
                            json.dump(scores, f, indent=2)

        if index != len(self.records):
            raise ValueError(f'Replay interface ended after {index} decisions; '
                             f'{len(self.records) - index} recorded decisions remain')
        log.info(f'Replayed all {index} decisions')
