from importlib import reload

def reload_all():
    # Useful function for developing in an interactive environment without having to restart the kernel
    
    from align_system.algorithms.lib import util
    from align_system.algorithms.lib import language_model as lm
    from align_system.algorithms.lib.chat import dialog_tokenizer as dt
    from align_system.algorithms.lib.chat import chat_language_model as clm
    from align_system.algorithms import llama_2_kdma_predicting_adm as kpa


    # Reload in the correct order
    for module in [util, lm, dt, clm, kpa]:
        reload(module)
