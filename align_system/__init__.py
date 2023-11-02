from importlib import reload

def reload_all():
    # Useful function for developing in an interactive environment without having to restart the kernel
    
    from align_system.algorithms.lib import util
    from align_system.algorithms.lib import language_model as lm
    from align_system.algorithms.lib import aligned_decision_maker as adm
    from align_system.algorithms.lib.chat import chat_language_model as clm
    
    from align_system.evaluation import adm_evaluator as adme
    
    from align_system.algorithms import chat_kdma_predicting_adm as kpa
    from align_system.algorithms import llama_2_single_kdma_adm as ska


    # Reload in the correct order
    for module in [util, lm, clm, adm, adme, kpa, ska]:
        reload(module)
