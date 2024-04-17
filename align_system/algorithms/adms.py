from align_system.algorithms.kaleido_adm import KaleidoADM
from align_system.algorithms.llama_2_single_kdma_adm import Llama2SingleKDMAADM
from align_system.algorithms.hybrid_kaleido_adm import HybridKaleidoADM
from align_system.algorithms.random_adm import RandomADM
from align_system.algorithms.oracle_adm import OracleADM

REGISTERED_ADMS = {
    'KaleidoADM': KaleidoADM,
    'HybridKaleidoADM': HybridKaleidoADM,
    'SingleKDMAADM': Llama2SingleKDMAADM,
    'RandomADM': RandomADM,
    'OracleADM': OracleADM,
}
