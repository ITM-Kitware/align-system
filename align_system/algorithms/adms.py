from align_system.algorithms.kaleido_adm import KaleidoADM
from align_system.algorithms.llama_2_single_kdma_adm import Llama2SingleKDMAADM

REGISTERED_ADMS = {
    'KaleidoADM': KaleidoADM,
    'SingleKDMAADM': Llama2SingleKDMAADM,
}
